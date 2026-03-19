"""
SAR Sensitivity Explorer — M7c Webapp

Input a SMILES → see which positions on the molecule are most sensitive
to structural modification, based on 25M matched molecular pairs from
50 ChEMBL targets.

Run:
    streamlit run webapp/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

# Ensure the project source is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webapp.predict import (
    predict_positions,
    sensitivity_to_label,
    PositionResult,
    FEATURE_NAMES,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SAR Sensitivity Explorer",
    page_icon="🧪",
    layout="wide",
)

# ── Load model metadata ─────────────────────────────────────────────────────

META_PATH = Path(__file__).parent / "model" / "model_meta.json"
with open(META_PATH) as f:
    MODEL_META = json.load(f)

# ── Example molecules ────────────────────────────────────────────────────────

EXAMPLES = {
    "Imatinib (BCR-ABL)": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "Erlotinib (EGFR)": "COCCOC1=CC2=C(C=C1OCCOC)C(=NC=N2)NC3=CC(=CC=C3)C#C",
    "Celecoxib (COX-2)": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
    "Atorvastatin (HMG-CoA)": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
    "Sorafenib (Multi-kinase)": "CNC(=O)C1=CC(=CC=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F",
    "Diclofenac (COX)": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "Simple aniline": "Nc1ccccc1",
    "4-Aminobiphenyl": "Nc1ccc(-c2ccccc2)cc1",
}

# ── Feature descriptions for chemists ────────────────────────────────────────

FEATURE_DESCRIPTIONS = {
    "ctx_n_donor_4A": ("H-bond donors within 4\u00c5", "Nearby NH/OH groups that can donate H-bonds"),
    "ctx_n_acceptor_4A": ("H-bond acceptors within 4\u00c5", "Nearby N/O/halogen atoms with lone pairs"),
    "ctx_n_hydrophobic_4A": ("Hydrophobic atoms within 4\u00c5", "Nearby aliphatic carbons (no heteroatom neighbors)"),
    "ctx_n_aromatic_4A": ("Aromatic atoms within 4\u00c5", "Nearby aromatic ring atoms"),
    "ctx_sasa_attach": ("Solvent-accessible surface area", "How exposed the attachment point is (\u00c5\u00b2)"),
    "ctx_gasteiger_charge": ("Gasteiger partial charge", "Electronic character at attachment (- = electron-rich, + = electron-poor)"),
    "ctx_n_rotbonds_2": ("Rotatable bonds nearby", "Flexibility within 2 bonds of attachment"),
    "ctx_is_aromatic_attach": ("Aromatic attachment", "Whether the attachment atom is in an aromatic ring"),
    "ctx_n_heavy_4A": ("Heavy atoms within 4\u00c5", "Steric crowding around the attachment point"),
    "core_n_heavy": ("Core heavy atom count", "Size of the scaffold (larger = less sensitive)"),
    "core_n_rings": ("Core ring count", "Ring complexity of the scaffold"),
}


# ── Visualization ────────────────────────────────────────────────────────────

def sensitivity_color(value: float, vmin: float, vmax: float) -> tuple[float, float, float]:
    """Map sensitivity to a blue→white→red color. Returns (r, g, b) in [0, 1]."""
    if vmax == vmin:
        return (0.85, 0.85, 0.85)
    t = (value - vmin) / (vmax - vmin)  # 0 = least sensitive, 1 = most
    t = max(0.0, min(1.0, t))
    # Blue (cold) → White → Red (hot)
    if t < 0.5:
        s = t * 2  # 0→1 over the blue-to-white range
        r = 0.2 + 0.8 * s
        g = 0.3 + 0.7 * s
        b = 0.9 + 0.1 * s
    else:
        s = (t - 0.5) * 2  # 0→1 over the white-to-red range
        r = 1.0
        g = 1.0 - 0.7 * s
        b = 1.0 - 0.8 * s
    return (r, g, b)


def draw_molecule_with_sensitivity(
    smiles: str,
    results: list[PositionResult],
    width: int = 700,
    height: int = 450,
    show_rank_labels: bool = True,
) -> str:
    """Draw molecule SVG with atoms colored by predicted sensitivity."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    # Build atom → sensitivity mapping
    atom_sens = {}
    atom_rank = {}  # atom_idx → rank (1-based)
    for i, r in enumerate(results):
        atom_sens[r.atom_idx] = r.sensitivity
        atom_rank[r.atom_idx] = i + 1

    if not atom_sens:
        return ""

    vmin = min(atom_sens.values())
    vmax = max(atom_sens.values())

    # Expand coloring: also lightly color the R-group side atom
    atom_colors = {}
    atom_radii = {}
    highlight_atoms = []

    for r in results:
        color = sensitivity_color(r.sensitivity, vmin, vmax)
        atom_colors[r.atom_idx] = color
        atom_radii[r.atom_idx] = 0.4
        highlight_atoms.append(r.atom_idx)

        # Lightly color the R-group neighbor too
        if r.neighbor_idx not in atom_colors:
            # Fade toward white
            nc = sensitivity_color(r.sensitivity, vmin, vmax)
            faded = tuple(0.5 + 0.5 * c for c in nc)
            atom_colors[r.neighbor_idx] = faded
            atom_radii[r.neighbor_idx] = 0.25
            highlight_atoms.append(r.neighbor_idx)

    # Also highlight the cut bonds
    highlight_bonds = []
    bond_colors = {}
    for r in results:
        bond = mol.GetBondBetweenAtoms(r.atom_idx, r.neighbor_idx)
        if bond is not None:
            bidx = bond.GetIdx()
            highlight_bonds.append(bidx)
            color = sensitivity_color(r.sensitivity, vmin, vmax)
            bond_colors[bidx] = color

    # Add rank labels as atom notes (shows "#1", "#2" etc. near each position)
    if show_rank_labels:
        for atom_idx, rank in atom_rank.items():
            mol.GetAtomWithIdx(atom_idx).SetProp("atomNote", f"#{rank}")

    # Draw
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.5
    opts.padding = 0.15
    opts.additionalAtomLabelPadding = 0.1
    opts.annotationFontScale = 0.6

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# ── Main app ─────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────
    st.title("SAR Sensitivity Explorer")
    st.markdown(
        "Predict which positions on a molecule are most sensitive to structural "
        "modification, based on **25 million matched molecular pairs** across "
        "50 ChEMBL targets."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("About the Model")
        st.markdown(f"""
**Training data:** {MODEL_META['n_training_rows']:,} position observations
across {MODEL_META['n_targets_trained_on']} protein targets

**Validation (leave-one-target-out):**
- NDCG@3 = {MODEL_META['validation_metrics']['ndcg_at_3']:.3f}
- Hit@1 = {MODEL_META['validation_metrics']['hit_at_1']:.0%}
- Spearman = {MODEL_META['validation_metrics']['spearman']:.3f}

**What it predicts:** mean |ΔpActivity| — the expected magnitude
of potency change when the R-group at this position is modified.
Higher values = this position is more SAR-sensitive.

**Key finding:** SAR sensitivity is governed by local 3D
pharmacophore context and is **target-agnostic** — the same
rules apply across kinases, GPCRs, proteases, etc.
        """)

        st.divider()
        st.subheader("Top Predictive Features")
        if "feature_importances" in MODEL_META:
            imps = MODEL_META["feature_importances"]
            for name, val in sorted(imps.items(), key=lambda x: -x[1])[:5]:
                short_name = FEATURE_DESCRIPTIONS.get(name, (name, ""))[0]
                st.markdown(f"- **{short_name}**: {val:.1%}")

        st.divider()
        st.markdown(
            "_Sensitivity = mean |ΔpActivity| across all known modifications "
            "at equivalent positions in ChEMBL. A sensitivity of 1.0 means "
            "R-group changes at this position shift potency by ~1 log unit "
            "(10-fold) on average._"
        )

    # ── Input ─────────────────────────────────────────────────────────────
    col_input, col_examples = st.columns([3, 1])

    with col_examples:
        st.markdown("**Example molecules**")
        for name, smi in EXAMPLES.items():
            if st.button(name, key=f"ex_{name}", use_container_width=True):
                st.session_state["smiles_input"] = smi

    with col_input:
        smiles = st.text_input(
            "Enter SMILES",
            value=st.session_state.get("smiles_input", ""),
            placeholder="e.g. CC1=C(C=C(C=C1)NC(=O)...",
            help="Paste a SMILES string for any drug-like molecule",
        )

    if not smiles:
        st.info("Enter a SMILES or click an example molecule to get started.")
        return

    # Validate
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES. Please check the input.")
        return

    # ── Predict ───────────────────────────────────────────────────────────
    with st.spinner("Generating conformer and computing features..."):
        results = predict_positions(smiles)

    if not results:
        st.warning(
            "No fragmentable positions found. The molecule may be too small "
            "or have no acyclic single bonds between heavy atoms."
        )
        return

    # ── Results ───────────────────────────────────────────────────────────
    col_mol, col_table = st.columns([1.2, 1])

    with col_mol:
        st.subheader("Position Sensitivity Map")
        svg = draw_molecule_with_sensitivity(smiles, results)
        if svg:
            st.image(svg, use_container_width=True)

        # Color legend
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-top:-10px;">'
            '<span style="color:#334de6;font-weight:bold;">● Low sensitivity</span>'
            '<span style="color:#999;">→</span>'
            '<span style="color:#e63333;font-weight:bold;">● High sensitivity</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_table:
        st.subheader("Position Ranking")

        for i, r in enumerate(results):
            label = sensitivity_to_label(r.sensitivity)

            # Color indicator
            if label in ("Very High", "High"):
                badge = "🔴"
            elif label == "Moderate":
                badge = "🟡"
            else:
                badge = "🔵"

            with st.expander(
                f"{badge} **#{i+1}** — Sensitivity: **{r.sensitivity:.2f}** ({label}) "
                f"| R-group: `{r.rgroup_smiles}`",
                expanded=(i < 3),
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Atom index:** {r.atom_idx}")
                    st.markdown(f"**Percentile:** {r.percentile:.0f}th")
                    st.markdown(f"**Cut bond:** atom {r.atom_idx} — atom {r.neighbor_idx}")
                with c2:
                    st.markdown(f"**Core size:** {r.features.get('core_n_heavy', 0):.0f} heavy atoms")
                    st.markdown(f"**Core rings:** {r.features.get('core_n_rings', 0):.0f}")

                # Feature breakdown
                st.markdown("**Feature breakdown:**")
                feat_rows = []
                for fname in FEATURE_NAMES:
                    val = r.features.get(fname, 0.0)
                    short, desc = FEATURE_DESCRIPTIONS.get(fname, (fname, ""))
                    feat_rows.append({"Feature": short, "Value": f"{val:.3f}", "Description": desc})

                # Show as a compact table
                import pandas as pd
                st.dataframe(
                    pd.DataFrame(feat_rows),
                    hide_index=True,
                    use_container_width=True,
                )

    # ── Interpretation panel ──────────────────────────────────────────────
    st.divider()
    st.subheader("Interpretation Guide")

    # Overall molecule summary
    sens_values = [r.sensitivity for r in results]
    spread = max(sens_values) - min(sens_values)

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.metric("Most sensitive position", f"{results[0].sensitivity:.2f}")
        st.markdown(f"R-group at this position: `{results[0].rgroup_smiles}`")
    with col_i2:
        st.metric("Position spread", f"{spread:.2f}")
        if spread > 0.3:
            st.markdown("Positions vary **meaningfully** — the model can distinguish SAR-sensitive vs. stable sites.")
        elif spread > 0.15:
            st.markdown("Moderate variation between positions. The top-ranked position is a reasonable starting point.")
        else:
            st.markdown("Positions are **similar** in predicted sensitivity. Consider other criteria (synthetic accessibility, IP) to prioritize.")

    st.markdown("""
**How to read the results:**
- **Sensitivity** is the predicted mean |ΔpActivity| — how much potency changes on average when the R-group at this position is swapped. A value of 1.0 means ~10-fold potency shifts are typical.
- **Higher sensitivity** = more SAR variation = good place to explore if you want to find potent analogs quickly.
- **Lower sensitivity** = SAR is flat = modifications here are less likely to dramatically change potency.
- The model captures **position sensitivity**, not direction — it tells you WHERE to invest synthesis, not whether a specific change will increase or decrease potency.

**What makes a position sensitive?**
The strongest predictors are scaffold simplicity (smaller cores with fewer rings) and solvent exposure (higher SASA, less steric crowding). Intuitively: when the R-group represents a larger fraction of the molecule's binding interaction, changes there have bigger effects on potency.
    """)


if __name__ == "__main__":
    main()
