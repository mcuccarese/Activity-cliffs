# M8: Interactive Explainability — Implementation Plan

> **Model:** Sonnet (implementation)
> **Estimated steps:** 4 phases, each independently testable
> **Pre-requisite reads:** `PROGRESS_LOG.md`, `webapp/app.py`, `webapp/predict.py`, `scripts/build_evidence_index.py`

---

## Overview

Three user requirements:
1. **Click on a bond to explore** — currently the molecule is a static SVG with no interaction
2. **Show structures** — currently evidence shows R-group SMILES as text (`R-OMe → R-Cl`), not actual molecular images
3. **Link to sources** — currently shows `EGFR (CHEMBL203)` as text, not clickable links to ChEMBL

---

## Phase 1: Clickable Position Explorer

### Problem
The molecule SVG is static. The position ranking is an expander list. There's no way to click a bond on the molecule and see its details.

### Solution
Redesign the layout so that positions are **selectable** and the detail panel updates based on selection.

### Implementation

**1a. Add position selector below the molecule**

In `webapp/app.py`, replace the current two-column layout (molecule | expander list) with:

```
┌─────────────────────────────────────────────────────┐
│              Molecule SVG (full width)               │
│         (positions labeled #1, #2, #3...)            │
└─────────────────────────────────────────────────────┘
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ #1 🔴│ │ #2 🔴│ │ #3 🟡│ │ #4 🔵│ │ #5 🔵│  ← clickable pills
└──────┘ └──────┘ └──────┘ └──────┘ └──────┘
┌─────────────────────────────────────────────────────┐
│           Detail panel for selected position         │
│  ┌──────────────┐  ┌──────────────────────────────┐ │
│  │  Features     │  │  Evidence (with structures)  │ │
│  └──────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

Use `st.pills` (Streamlit ≥1.37) or `st.radio` with `horizontal=True` for the position selector. Each pill shows: `#1 — 1.23 (High)` with color coding.

Store selected position in `st.session_state["selected_position"]`.

**1b. Re-render SVG to highlight selected bond**

When a position is selected, re-draw the molecule SVG with:
- ALL positions still colored by sensitivity (same as now)
- The SELECTED position's bond drawn with a **thick black outline** or **dashed circle** to distinguish it from the rest
- Increase the selected atom's radius from 0.4 → 0.6

Modify `draw_molecule_with_sensitivity()` to accept an optional `selected_idx: int | None` parameter. When set, emphasize that position visually.

**1c. Detail panel for the selected position**

Replace the expander list with a single detail panel that shows:
- Left column: Feature breakdown table (same data as current popover)
- Right column: Evidence from ChEMBL (see Phase 2 for structure rendering)

The key change is going from "all positions expanded" to "one position at a time, selected interactively."

### Files to modify
- `webapp/app.py`: Layout restructure, add pills selector, detail panel

---

## Phase 2: Render Evidence Structures

### Problem
Evidence currently shows R-group SMILES as monospace text: `R-OMe → R-Cl`. A medicinal chemist wants to SEE the structures.

### Solution
Render each evidence MMP as a pair of small molecule SVGs showing the full molecules (or at minimum the R-groups) with the modification highlighted.

### Implementation

**2a. Add structure rendering utility**

Create a helper function in `webapp/app.py`:

```python
def render_rgroup_pair_svg(rgroup_from: str, rgroup_to: str, width=150, height=100) -> tuple[str, str]:
    """Render two R-group structures as small SVGs."""
    # Parse R-group SMILES (replace [*:1] with H for rendering, or keep as dummy)
    # Use rdMolDraw2D.MolDraw2DSVG for each
    # Return (svg_from, svg_to)
```

For each evidence example, display:

```
┌────────────┐      ┌────────────┐
│  [mol_from] │  →   │  [mol_to]  │   ΔpActivity = +1.23
│  (structure)│      │ (structure) │   EGFR (link)
└────────────┘      └────────────┘
```

Use `st.columns([1, 0.3, 1, 1.5])` for the layout.

**2b. Full molecule rendering (if SMILES available)**

The evidence index currently stores `rgroup_from` and `rgroup_to` SMILES. For Phase 2, also store `smiles_from` and `smiles_to` (full molecule SMILES) in the evidence index (see Phase 3).

When full molecule SMILES are available, render the complete molecules side-by-side with the modified region highlighted. To highlight the R-group region:
1. Parse both molecules
2. Find the maximum common substructure (MCS) or use the core SMILES as a substructure query
3. Highlight atoms NOT in the core (= the R-group atoms) in the drawing

**If full SMILES aren't available** (fallback), render just the R-groups. Replace `[*:1]` with a methyl group for rendering context, or render as-is with the dummy atom.

**2c. Practical R-group rendering approach**

For the MVP, the simplest approach that looks good:
1. Take `rgroup_from` SMILES (e.g., `[*:1]OC`)
2. Replace `[*:1]` with a colored atom label or keep the dummy atom
3. Render with `MolDraw2DSVG` at 120×90 px
4. Embed as inline SVG in the evidence card via `st.markdown(unsafe_allow_html=True)` or `st.image()`

RDKit can render molecules with `[*]` atoms — they show as a star. This is actually standard medchem notation for the attachment point, so it works well.

### Files to modify
- `webapp/app.py`: Add `render_rgroup_pair_svg()`, update `_render_evidence()`

---

## Phase 3: Enrich Evidence Index with Compound IDs and Full SMILES

### Problem
The evidence index currently stores only: `target_id, target_name, rgroup_from, rgroup_to, delta_pActivity, abs_delta`. Missing: compound ChEMBL IDs and full molecule SMILES needed for linkouts and structure rendering.

### Solution
Modify `scripts/build_evidence_index.py` to also store `smiles_from`, `smiles_to`, and `molecule_chembl_id` for each evidence MMP.

### Implementation

**3a. Get compound ChEMBL IDs from the database**

The MMP parquet has `mol_from` and `mol_to` which are `molregno` values from ChEMBL. To get compound ChEMBL IDs:

```python
# Query ChEMBL SQLite for molregno → molecule_chembl_id mapping
sql = "SELECT molregno, chembl_id AS molecule_chembl_id FROM molecule_dictionary"
molregno_map = pd.read_sql(sql, conn).set_index("molregno")["molecule_chembl_id"].to_dict()
```

Add `--chembl-sqlite` option to `build_evidence_index.py`.

**3b. Store additional fields in evidence lookup**

Expand each evidence record to include:

```python
{
    "target_id": "CHEMBL203",
    "target_name": "EGFR",
    "rgroup_from": "[*:1]OC",
    "rgroup_to": "[*:1]Cl",
    "smiles_from": "COc1ccc(NC(=O)...)cc1",      # NEW
    "smiles_to": "Clc1ccc(NC(=O)...)cc1",         # NEW
    "molecule_chembl_id_from": "CHEMBL123456",     # NEW
    "molecule_chembl_id_to": "CHEMBL789012",       # NEW
    "delta_pActivity": 1.23,
    "abs_delta": 1.23,
}
```

Load from all_mmps.parquet — the columns `smiles_from`, `smiles_to`, `mol_from`, `mol_to` are already there.

**3c. Update EvidenceExample dataclass**

In `webapp/predict.py`, add fields:

```python
@dataclass
class EvidenceExample:
    target_id: str
    target_name: str
    rgroup_from: str
    rgroup_to: str
    delta_pActivity: float
    abs_delta: float
    similarity: float
    source: str
    # NEW fields:
    smiles_from: str = ""
    smiles_to: str = ""
    molecule_chembl_id_from: str = ""
    molecule_chembl_id_to: str = ""
```

### Files to modify
- `scripts/build_evidence_index.py`: Add ChEMBL lookup, store extra fields
- `webapp/predict.py`: Update `EvidenceExample` dataclass, update `find_evidence()` to populate new fields

### Rebuild command
```bash
python scripts/build_evidence_index.py --chembl-sqlite "D:\Mike project data\Activity cliffs\chembl_36\chembl_36_sqlite\chembl_36.db"
```

---

## Phase 4: ChEMBL Linkouts

### Problem
Evidence shows target names as plain text. No way to look up the source data.

### Solution
Add hyperlinks to ChEMBL for targets AND individual compounds.

### Implementation

**4a. Target links**

Replace plain text `EGFR (CHEMBL203)` with a clickable link:
```html
<a href="https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL203" target="_blank">
  EGFR (CHEMBL203)
</a>
```

**4b. Compound links**

For each evidence MMP, add links to the two compounds:
```html
<a href="https://www.ebi.ac.uk/chembl/compound_report_card/CHEMBL123456" target="_blank">
  CHEMBL123456
</a>
```

**4c. Updated evidence card layout**

Each evidence card becomes:

```
┌─────────────────────────────────────────────────────────────────┐
│ EGFR (CHEMBL203) ↗                           ΔpActivity = +1.23│
│                                                                 │
│ ┌──────────────┐         ┌──────────────┐                       │
│ │  [structure]  │   →     │  [structure]  │                      │
│ │  CHEMBL123 ↗  │         │  CHEMBL456 ↗  │                      │
│ └──────────────┘         └──────────────┘                       │
│                                                                 │
│ 87% pharmacophore similarity                                    │
└─────────────────────────────────────────────────────────────────┘
```

Where ↗ indicates external links that open in a new tab.

### Files to modify
- `webapp/app.py`: Update `_render_evidence()` with links and structure SVGs

---

## Execution Order

1. **Phase 3 first** (enrich evidence index) — this rebuilds the data that Phases 2 and 4 need
2. **Phase 1** (clickable position explorer) — layout redesign, can work with current evidence rendering temporarily
3. **Phase 2** (render structures) — uses the new `smiles_from`/`smiles_to` from Phase 3
4. **Phase 4** (linkouts) — uses the new `molecule_chembl_id` from Phase 3

### Prompt for Sonnet

> Read `M8_INTERACTIVE_EXPLAINABILITY_PLAN.md`, `webapp/app.py`, `webapp/predict.py`, and `scripts/build_evidence_index.py`. Then implement M8 in this order:
>
> **Phase 3** (data enrichment): Modify `build_evidence_index.py` to also store `smiles_from`, `smiles_to`, and compound ChEMBL IDs (`molecule_chembl_id_from/to`) from the ChEMBL SQLite database. Add `--chembl-sqlite` CLI option. Update `EvidenceExample` dataclass in `predict.py`. Then rebuild the evidence index.
>
> **Phase 1** (clickable bonds): Redesign `app.py` layout so the molecule SVG is full-width at top, position selection is via `st.pills` or horizontal radio buttons below it, and a single detail panel shows the selected position. Modify `draw_molecule_with_sensitivity` to accept a `selected_idx` that gets extra visual emphasis.
>
> **Phase 2** (structure rendering): Add a function to render R-group pairs as small inline SVGs using RDKit's MolDraw2DSVG. Show side-by-side "before → after" structures in each evidence card. If full molecule SMILES are available, render those; otherwise render R-groups with [*] attachment point.
>
> **Phase 4** (linkouts): Add hyperlinks to ChEMBL target pages (`https://www.ebi.ac.uk/chembl/target_report_card/CHEMBLXXX`) and compound pages (`https://www.ebi.ac.uk/chembl/compound_report_card/CHEMBLXXX`). Links open in new tabs.
>
> Key file paths:
> - ChEMBL SQLite: `D:\Mike project data\Activity cliffs\chembl_36\chembl_36_sqlite\chembl_36.db`
> - MMP data: `outputs/mmps/all_mmps.parquet` (columns: mol_from, mol_to, smiles_from, smiles_to, core_smiles, rgroup_from, rgroup_to, target_chembl_id, delta_pActivity, abs_delta_pActivity)
> - `mol_from`/`mol_to` are `molregno` values — join to `molecule_dictionary.molregno` → `molecule_dictionary.chembl_id` in ChEMBL SQLite
> - Current evidence index: `webapp/model/evidence_index.pkl`
> - 3D context features: `outputs/features/context_3d.parquet`
>
> Test by running `streamlit run webapp/app.py` after each phase.

---

## Technical Notes

### Streamlit version check
Run `pip show streamlit` — if < 1.37, `st.pills` won't be available. Fall back to `st.radio(horizontal=True)` or `st.segmented_control`.

### SVG rendering for R-groups
```python
from rdkit.Chem.Draw import rdMolDraw2D

def rgroup_to_svg(smiles: str, w=150, h=100) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    drawer.drawOptions().padding = 0.2
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
```

R-group SMILES like `[*:1]OC` render fine in RDKit — the `[*]` shows as a star/dot, which is standard medchem notation for an attachment point.

### ChEMBL URL patterns
- Target: `https://www.ebi.ac.uk/chembl/target_report_card/{target_chembl_id}`
- Compound: `https://www.ebi.ac.uk/chembl/compound_report_card/{molecule_chembl_id}`
- These are stable public URLs that don't require authentication.

### Evidence index size
Adding `smiles_from`, `smiles_to`, and two ChEMBL IDs per evidence record will increase the pickle size. Current: ~50 MB. Expected: ~55-60 MB. Acceptable.

### Backward compatibility
The new evidence index adds fields but doesn't remove any. `find_evidence()` should handle missing fields gracefully (default to empty string) so the app doesn't crash if someone runs it with an old index.
