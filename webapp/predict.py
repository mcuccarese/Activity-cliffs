"""
Prediction pipeline: SMILES → fragment → 3D features → position sensitivity.

Takes any input molecule, finds all single-cut fragmentable bonds,
computes 3D pharmacophore context at each attachment point, and predicts
SAR sensitivity per position using the trained HGB model.

Also provides evidence lookup: for each predicted position, find real
ChEMBL examples with similar pharmacophore context to explain the prediction.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMMPA

# Import the 3D context feature computation from the main package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from activity_cliffs.features.context_3d import compute_3d_context, CONTEXT_3D_FEATURES


MODEL_PATH = Path(__file__).parent / "model" / "position_hgb.pkl"
EVIDENCE_INDEX_PATH = Path(__file__).parent / "model" / "evidence_index.pkl"

# Feature order must match training data
FEATURE_NAMES = [
    f"ctx_{c}" for c in CONTEXT_3D_FEATURES
] + ["core_n_heavy", "core_n_rings"]


@dataclass
class EvidenceExample:
    """A real MMP from ChEMBL that supports a prediction."""
    target_id: str           # e.g. "CHEMBL203"
    target_name: str         # e.g. "EGFR"
    rgroup_from: str         # R-group SMILES before change
    rgroup_to: str           # R-group SMILES after change
    delta_pActivity: float   # signed potency change
    abs_delta: float         # |ΔpActivity|
    similarity: float        # cosine similarity to query position (0-1)
    source: str              # "exact" if same core, "similar" if neighbor


@dataclass
class PositionResult:
    """Result for one fragmentable position on the input molecule."""
    atom_idx: int            # atom index in original mol (core-side of cut bond)
    neighbor_idx: int        # atom index on the R-group side of the cut bond
    bond_idx: int            # bond index of the cut bond
    core_smiles: str         # core SMILES with [*:1] at cut
    rgroup_smiles: str       # R-group fragment SMILES
    sensitivity: float       # predicted mean |ΔpActivity| if this position is modified
    percentile: float        # percentile vs training data (0-100)
    features: dict[str, float] = field(default_factory=dict)
    evidence: list[EvidenceExample] = field(default_factory=list)


def _load_model():
    """Load the trained HGB model."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def _core_n_heavy(smi: str) -> int:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 0)


def _core_n_rings(smi: str) -> int:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return mol.GetRingInfo().NumRings()


def _find_fragmentable_bonds(mol: Chem.Mol) -> list[tuple[int, int, int]]:
    """
    Find all single acyclic bonds between two heavy atoms.

    Returns list of (bond_idx, begin_atom_idx, end_atom_idx).
    Excludes bonds to H and bonds in rings.
    """
    results = []
    ri = mol.GetRingInfo()
    for bond in mol.GetBonds():
        # Must be single, non-ring
        if bond.GetBondTypeAsDouble() != 1.0:
            continue
        if bond.IsInRing():
            continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        # Both must be heavy atoms
        if a1.GetAtomicNum() < 2 or a2.GetAtomicNum() < 2:
            continue
        # Skip if either atom has only 1 heavy neighbor (terminal atom —
        # cutting here would give a single-atom R-group like F, Cl, etc.
        # which is fine, we keep these)
        results.append((bond.GetIdx(), a1.GetIdx(), a2.GetIdx()))
    return results


def _fragment_at_bond(
    mol: Chem.Mol, bond_idx: int, a1_idx: int, a2_idx: int
) -> list[tuple[str, str, int, int]]:
    """
    Fragment molecule at a specific bond.

    Returns list of (core_smiles, rgroup_smiles, core_atom_idx, rgroup_atom_idx)
    where core is the LARGER fragment.
    """
    em = Chem.RWMol(mol)
    em.RemoveBond(a1_idx, a2_idx)

    # Add dummy atoms with DIFFERENT map numbers to track which is which
    dummy1_idx = em.AddAtom(Chem.Atom(0))  # [*:1] bonded to a1
    em.AddBond(a1_idx, dummy1_idx, Chem.BondType.SINGLE)
    em.GetAtomWithIdx(dummy1_idx).SetAtomMapNum(1)

    dummy2_idx = em.AddAtom(Chem.Atom(0))  # [*:2] bonded to a2
    em.AddBond(a2_idx, dummy2_idx, Chem.BondType.SINGLE)
    em.GetAtomWithIdx(dummy2_idx).SetAtomMapNum(2)

    try:
        frags = Chem.GetMolFrags(em.GetMol(), asMols=True, sanitizeFrags=True)
    except Exception:
        return []

    if len(frags) != 2:
        return []

    # Identify which fragment contains a1 (has dummy with map=1)
    # and which contains a2 (has dummy with map=2)
    def has_map_num(frag_mol, map_num):
        return any(a.GetAtomMapNum() == map_num for a in frag_mol.GetAtoms())

    def heavy_count(m):
        return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 0)

    if has_map_num(frags[0], 1):
        frag_a1, frag_a2 = frags[0], frags[1]
    else:
        frag_a1, frag_a2 = frags[1], frags[0]

    hc_a1 = heavy_count(frag_a1)
    hc_a2 = heavy_count(frag_a2)

    # Normalize: set all dummy map numbers to 1 for canonical core SMILES
    def set_dummy_map(frag_mol, target_map=1):
        for a in frag_mol.GetAtoms():
            if a.GetAtomicNum() == 0:
                a.SetAtomMapNum(target_map)
        return Chem.MolToSmiles(frag_mol)

    # The LARGER fragment is the core, smaller is R-group
    if hc_a1 >= hc_a2:
        core_smi = set_dummy_map(frag_a1)
        rg_smi = set_dummy_map(frag_a2)
        core_atom, rg_atom = a1_idx, a2_idx
    else:
        core_smi = set_dummy_map(frag_a2)
        rg_smi = set_dummy_map(frag_a1)
        core_atom, rg_atom = a2_idx, a1_idx

    return [(core_smi, rg_smi, core_atom, rg_atom)]


def predict_positions(smiles: str) -> list[PositionResult]:
    """
    Main prediction function: SMILES → list of PositionResult.

    Each result represents one fragmentable position on the molecule,
    with predicted SAR sensitivity.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    mol = Chem.AddHs(mol)
    mol = Chem.RemoveHs(mol)  # normalize

    model = get_model()

    # Find all fragmentable bonds
    bonds = _find_fragmentable_bonds(mol)
    if not bonds:
        return []

    # For each bond, fragment and compute features
    results = []
    seen_cores = set()

    for bond_idx, a1, a2 in bonds:
        frags = _fragment_at_bond(mol, bond_idx, a1, a2)
        for core_smi, rg_smi, core_atom, rg_atom in frags:
            # Deduplicate: same core = same position
            if core_smi in seen_cores:
                continue
            seen_cores.add(core_smi)

            # Compute 3D context features
            ctx_feats = compute_3d_context(core_smi)

            # Core topology features
            n_heavy = _core_n_heavy(core_smi)
            n_rings = _core_n_rings(core_smi)

            # Build feature vector (must match training order)
            x = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            x[:len(ctx_feats)] = ctx_feats
            x[-2] = float(n_heavy)
            x[-1] = float(n_rings)

            # Predict
            pred = model.predict(x.reshape(1, -1))[0]
            pred = max(0.0, pred)  # sensitivity can't be negative

            # Feature dict for interpretability
            feat_dict = dict(zip(FEATURE_NAMES, [float(v) for v in x]))

            results.append(PositionResult(
                atom_idx=core_atom,
                neighbor_idx=rg_atom,
                bond_idx=bond_idx,
                core_smiles=core_smi,
                rgroup_smiles=rg_smi,
                sensitivity=float(pred),
                percentile=0.0,  # filled below
                features=feat_dict,
            ))

    if not results:
        return []

    # Compute percentiles relative to training data distribution
    # Training: mean=0.834, std=0.434, p25=0.537, p75=1.023, p90=1.378
    from scipy import stats
    training_mean = 0.834
    training_std = 0.434
    for r in results:
        # Use a normal approximation for percentile
        z = (r.sensitivity - training_mean) / training_std
        r.percentile = float(stats.norm.cdf(z) * 100)

    # Sort by sensitivity (highest first)
    results.sort(key=lambda r: r.sensitivity, reverse=True)

    # Attach evidence examples from real ChEMBL MMPs
    for r in results:
        x = np.array([r.features[fn] for fn in FEATURE_NAMES], dtype=np.float32)
        r.evidence = find_evidence(x, r.core_smiles)

    return results


def sensitivity_to_label(sensitivity: float) -> str:
    """Human-readable sensitivity label."""
    if sensitivity >= 1.5:
        return "Very High"
    elif sensitivity >= 1.0:
        return "High"
    elif sensitivity >= 0.7:
        return "Moderate"
    elif sensitivity >= 0.4:
        return "Low"
    else:
        return "Very Low"


# ── Evidence index ───────────────────────────────────────────────────────

_EVIDENCE_INDEX = None


def _load_evidence_index():
    """Load the pre-built evidence index (BallTree + evidence lookup)."""
    if not EVIDENCE_INDEX_PATH.exists():
        return None
    with open(EVIDENCE_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def get_evidence_index():
    global _EVIDENCE_INDEX
    if _EVIDENCE_INDEX is None:
        _EVIDENCE_INDEX = _load_evidence_index()
    return _EVIDENCE_INDEX


def find_evidence(
    features: np.ndarray,
    core_smiles: str,
    k_neighbors: int = 10,
    max_examples: int = 8,
) -> list[EvidenceExample]:
    """
    Find real MMP evidence for a predicted position.

    Strategy:
    1. Check if the exact core_smiles exists in the evidence index (exact match)
    2. Find k nearest cores by pharmacophore context features (similar match)
    3. Combine and deduplicate, prioritizing exact matches and largest |delta|

    Args:
        features: 11-dim feature vector for this position (same order as training)
        core_smiles: the core SMILES from fragmentation
        k_neighbors: number of nearest cores to retrieve
        max_examples: max evidence examples to return

    Returns:
        List of EvidenceExample sorted by |delta| descending
    """
    idx = get_evidence_index()
    if idx is None:
        return []

    evidence_lookup = idx["evidence_lookup"]
    scaler = idx["scaler"]
    tree = idx["tree"]
    examples: list[EvidenceExample] = []
    seen: set[tuple[str, str, str]] = set()  # (target, rg_from, rg_to)

    # 1. Exact match: same core exists in ChEMBL
    if core_smiles in evidence_lookup:
        for ex in evidence_lookup[core_smiles]:
            key = (ex["target_id"], ex["rgroup_from"], ex["rgroup_to"])
            if key not in seen:
                seen.add(key)
                examples.append(EvidenceExample(
                    target_id=ex["target_id"],
                    target_name=ex["target_name"],
                    rgroup_from=ex["rgroup_from"],
                    rgroup_to=ex["rgroup_to"],
                    delta_pActivity=ex["delta_pActivity"],
                    abs_delta=ex["abs_delta"],
                    similarity=1.0,
                    source="exact",
                ))

    # 2. Nearest-neighbor: find cores with similar pharmacophore context
    x_scaled = scaler.transform(features.reshape(1, -1))
    dists, idxs = tree.query(x_scaled, k=k_neighbors)

    core_smiles_arr = idx["core_smiles"]
    for dist, neighbor_idx in zip(dists[0], idxs[0]):
        neighbor_core = str(core_smiles_arr[neighbor_idx])
        if neighbor_core == core_smiles:
            continue  # already handled as exact match

        # Convert distance to similarity (0-1 scale, 1 = identical)
        similarity = 1.0 / (1.0 + dist)

        if neighbor_core in evidence_lookup:
            for ex in evidence_lookup[neighbor_core]:
                key = (ex["target_id"], ex["rgroup_from"], ex["rgroup_to"])
                if key not in seen:
                    seen.add(key)
                    examples.append(EvidenceExample(
                        target_id=ex["target_id"],
                        target_name=ex["target_name"],
                        rgroup_from=ex["rgroup_from"],
                        rgroup_to=ex["rgroup_to"],
                        delta_pActivity=ex["delta_pActivity"],
                        abs_delta=ex["abs_delta"],
                        similarity=similarity,
                        source="similar",
                    ))

    # Sort: exact matches first, then by |delta| descending
    examples.sort(key=lambda e: (0 if e.source == "exact" else 1, -e.abs_delta))

    return examples[:max_examples]
