# Rigorous Review & Out-of-Distribution Testing Plan

> **Purpose:** Anticipate expert critiques before public release. Structure experiments
> that demonstrate generalizability and surface limitations with clarity.

---

## RESULTS STATUS (2026-03-21)

**7 of 11 experiments completed.** Full results in `outputs/ood/OOD_RESULTS_SUMMARY.md`.

| # | Experiment | Status | Key Result |
|---|---|---|---|
| 1 | Feature ablation | DONE | -core_n_heavy heuristic (0.966) **beats** full HGB (0.959) |
| 2 | Novel scaffold holdout | DONE | Seen 0.965 vs unseen 0.945; heuristic still wins |
| 3 | Target family holdout | DONE | All 6 families: heuristic beats HGB; mean 0.943 |
| 4 | Temporal split | DONE | Train<=2015, test>2015: heuristic 0.948, HGB 0.936 |
| 6 | Learning curve | DONE | 5% data = 100% data (model is data-free) |
| 8 | Directional analysis | DONE | 64% neutral, sensitivity independent of direction |
| 9 | Permutation tests | DONE | p=0.000, Cohen's d=346 for NDCG; 50/50 targets significant |
| 5 | Feature sensitivity | NOT STARTED | |
| 7 | External validation | NOT STARTED | |
| 10 | Calibration/uncertainty | NOT STARTED | |
| 11 | Hyperparam sensitivity | NOT STARTED | |

**Headline finding:** The ML model adds zero NDCG value over a trivial
`-core_n_heavy` heuristic. 3D features improve Spearman by +0.02 only.
The *position-level reframe* and *universality of the core-size rule* are
the genuine contributions.

---

## Part 1: Predicted Expert Critiques

### A. Medicinal Chemist Critiques

**A1. "Sensitivity ≠ actionability — predicting magnitude without direction is incomplete"**

The position model predicts mean |ΔpActivity| — how much potency *changes*, not
whether it goes up or down. A position could score "Very High" because every
modification there *destroys* activity. A medchemist needs to know: "modify here
to *improve* potency," not just "things happen here."

- **Severity:** Moderate. Partially mitigated by the change-type recommendations
  (M9) and evidence panel, but the core metric is unsigned.
- **OOD test:** Experiment 8 (directional analysis) below.
- **Honest framing:** Acknowledge this explicitly. The tool identifies *where SAR
  is steep* — the chemist still uses judgment on direction. This is consistent
  with the original Topliss tree (which also doesn't predict potency direction,
  just which substitution to try first to *learn the most*).

**A2. "Core size dominance (45%) is trivially obvious"**

Any medchemist knows that on a small fragment, the R-group dominates binding; on
a 500 Da scaffold, one methyl barely matters. If `core_n_heavy` explains most of
the variance, the model may just be an expensive way to say "smaller scaffolds
have more sensitive positions."

- **Severity:** High. This is the single most damaging critique.
- **OOD test:** Experiment 1 (ablation study) — quantify exact NDCG@3 with
  topology-only (2 features) vs full model (11 features). If the gap is <0.01,
  the 3D features are decorative.
- **Expected defense:** Permutation importance shows 3D features contribute ~47%
  collectively (charge 14%, SASA 11%, crowding 5%, aromatics 5%, etc.). But
  permutation importance ≠ incremental predictive gain. The ablation will
  settle this.

**A3. "Single-cut MMPs miss cooperative/multi-position SAR"**

Real lead optimization involves simultaneous modifications at 2–3 positions.
Synergy and antagonism between positions (e.g., para-Cl + meta-F together) are
invisible to single-cut MMPs. The model assumes positions are independent.

- **Severity:** Moderate. This is a genuine limitation of the data representation.
- **Framing:** Acknowledge as a scope boundary. Single-cut MMPs are the
  methodological standard (Hussain & Rea, 2010). Multi-cut MMPs exist but are
  exponentially noisier. Future work: double-cut analysis.

**A4. "Validation on known drugs is circular"**

Imatinib, erlotinib, celecoxib, and diclofenac are all in ChEMBL. Their MMPs are
in the training data. Showing the model "correctly" ranks their positions is not
validation — it's memorization.

- **Severity:** High. The qualitative validation is illustrative, not evidential.
- **OOD test:** Experiments 2 (novel scaffolds), 4 (temporal split), and 7
  (external validation) address this directly.

**A5. "Change-type recommendations are too coarse for real decisions"**

"Try EWG" is less useful than "try 3-chloro at the para position." The Topliss
tree gives specific substituent recommendations; this model gives property-axis
recommendations.

- **Severity:** Low-moderate. The evidence panel partially compensates by showing
  real ChEMBL examples. Coarse recommendations are honest given Spearman=0.27.
- **Framing:** Position this as a triage tool (narrow from 11 axes to top 3),
  not a design tool. Specific R-group retrieval is listed as future work.

**A6. "The 0.268 Spearman for change-type doesn't inspire confidence"**

For the transformation-level model, 93% of variance is unexplained. A medchemist
might reasonably ask: "Why should I trust these rankings?"

- **Severity:** High for the change-type model specifically.
- **OOD test:** Experiment 9 (null model permutation test) — show that 0.268 is
  statistically far from zero. Experiment 6 (per-family breakdown) — show which
  target classes perform best/worst.
- **Framing:** Be candid: "This model is weakly predictive without protein
  features. It captures ligand-side signal only. Protein-aware models are the
  path to higher correlation." The value is in *relative* ranking of change types
  at a given position, not absolute magnitude prediction.

---

### B. Computational Chemist Critiques

**B1. "Single ETKDG conformer — not the bioactive conformation"**

The model uses one RDKit-generated conformer per core. The bioactive conformation
may differ substantially, especially for flexible molecules. Features like SASA
and pharmacophore counts at 4Å are conformation-dependent.

- **Severity:** Moderate. Partially mitigated by the fact that the model is
  trained AND evaluated on the same conformer method, so bias is systematic.
- **OOD test:** Experiment 5 (conformer sensitivity) — generate 10 conformers,
  compute features on each, measure prediction variance.
- **Defense:** The model learns *correlates* of position sensitivity, not physics.
  If ETKDG conformer features are consistently correlated with sensitivity across
  598K observations, the model captures that signal regardless of whether the
  conformer is bioactive. The systematic bias is baked in.

**B2. "4Å pharmacophore radius is arbitrary"**

Why 4Å and not 3.5Å or 5Å? This is a hard-coded hyperparameter that affects
all pharmacophore counts.

- **Severity:** Low.
- **OOD test:** Experiment 5 (hyperparameter sensitivity) — recompute features at
  3Å, 3.5Å, 4.5Å, 5Å, 6Å and re-evaluate.

**B3. "Gasteiger charges are crude — use AM1-BCC or xTB"**

Gasteiger partial charges are fast but inaccurate, especially for heterocycles
and charged species. Modern methods (AM1-BCC, GFN2-xTB) are far more reliable.

- **Severity:** Low-moderate. Gasteiger charge is the 2nd most important feature
  (14.1%). If it's a noisy proxy, better charges might improve the model.
- **OOD test:** Experiment 5 — recompute with AM1-BCC (via OpenBabel or ORCA)
  for a subset and compare. Pragmatic answer: Gasteiger is available everywhere
  (pure RDKit), enabling deployment without external dependencies.

**B4. "Free-ligand SASA ≠ bound-state SASA"**

Solvent accessibility on the isolated ligand has limited relevance to the binding
pocket geometry. A solvent-exposed position in free ligand may be deeply buried
in the protein.

- **Severity:** Moderate conceptually, but the model works empirically.
- **Defense:** The model doesn't claim to model binding physics. SASA_free is a
  proxy for intrinsic steric accessibility of the attachment point. It correlates
  with sensitivity because sterically accessible positions allow larger R-group
  diversity → more SAR variation. The correlation is statistical, not mechanistic.

**B5. "No consideration of solvation/desolvation penalties"**

Desolvation cost is critical for polar modifications — adding an H-bond donor in
a hydrophobic pocket costs ~5 kJ/mol in desolvation. The model ignores this.

- **Severity:** Low for position model (which predicts sensitivity, not direction).
  Moderate for change-type model (where polar vs lipophilic recommendations depend
  on desolvation context).
- **Framing:** Ligand-only model by design. Desolvation requires protein structure.

---

### C. ML Engineer Critiques

**C1. "NDCG@3 = 0.964 looks suspiciously high — what are the trivial baselines?"**

The position_ceiling.py script tests heuristic baselines (raw feature sorting)
and topology-only HGB, but the exact numbers aren't prominently reported.
A skeptic will ask: "How much of 0.964 comes from the model vs the easy
structure of the ranking task?"

- **Severity:** High. Must report all baselines prominently.
- **OOD test:** Experiment 1 (full ablation) — report: random, constant,
  core_n_heavy heuristic, topology HGB (2 feat), 3D context HGB (9 feat),
  full HGB (11 feat). Show the incremental gain at each step.

**C2. "No uncertainty quantification — point estimates are dangerous"**

The model outputs a single sensitivity score with no confidence interval. For
novel scaffolds far from training distribution, predictions could be arbitrarily
wrong with no indication.

- **Severity:** Moderate-high for a public tool.
- **OOD test:** Experiment 10 (calibration + uncertainty). Options: quantile
  regression HGB, conformal prediction, or bootstrap ensemble.

**C3. "Leave-one-target-out may overestimate generalization"**

All 50 targets are data-rich ChEMBL targets with >4,800 activities each. What
about targets with 50 compounds? Targets from novel protein families? The
evaluation set is biased toward well-studied targets.

- **Severity:** Moderate.
- **OOD test:** Experiment 3 (novel target families) and Experiment 6
  (low-data regime simulation).

**C4. "No temporal validation — the gold standard for drug discovery ML"**

Molecules in ChEMBL span decades. Training on all data and evaluating by
LOO-target conflates old and new chemistry. A model trained on pre-2020 data
and tested on 2020+ data would be a much stronger generalization claim.

- **Severity:** High. This is standard practice in CADD validation (Wallach & Heifets, 2018; Yang et al., 2019).
- **OOD test:** Experiment 4 (temporal split).

**C5. "Cross-target scaffold overlap inflates LOO-target results"**

The same core SMILES (e.g., a common benzimidazole) can appear in kinase data
(training) and GPCR data (test). The model has seen features for that core
during training — it's not truly novel.

- **Severity:** Moderate.
- **OOD test:** Experiment 2 (novel scaffold holdout) — explicitly hold out
  cores never seen in any training target.

**C6. "Hyperparameters not tuned — are you leaving performance on the table?"**

max_iter=300, max_depth=6, lr=0.1, min_leaf=50 are reasonable defaults, but
without a search, it's unknown whether tuning would change the story.

- **Severity:** Low. The ceiling tests (deeper HGB) showed marginal gains.
- **OOD test:** Experiment 11 (hyperparameter sensitivity) — random search
  over a modest grid, report variance of NDCG across configurations.

**C7. "The target homogeneity finding (r>0.91) is either profound or a red flag"**

If all 50 targets produce identical SAR patterns, either: (a) position-level
sensitivity is truly universal chemistry (the optimistic read), or (b) the
features are too coarse to distinguish targets (the pessimistic read — you're
averaging over signal).

- **Severity:** Moderate. Needs careful framing.
- **OOD test:** Experiment 3 (novel target families) tests whether the finding
  generalizes beyond the 50 selected targets.

---

## Part 2: Out-of-Distribution Testing Plan

### Experiment 1: Feature Ablation Study
**Goal:** Quantify incremental value of each feature group. Settle the
"core_n_heavy does all the work" critique.

**Protocol:**
1. LOO-target NDCG@3 for each configuration:
   - (a) Random baseline
   - (b) Global mean (constant prediction)
   - (c) `core_n_heavy` only — single feature, HGB
   - (d) `core_n_heavy + core_n_rings` — topology only, HGB (2 feat)
   - (e) 3D context only — HGB (9 feat, no topology)
   - (f) Full model — HGB (11 feat)
   - (g) Full model minus `core_n_heavy` (10 feat)
   - (h) Full model minus `gasteiger_charge` (10 feat)
2. Report NDCG@3, Hit@1, Spearman for each
3. Compute **incremental lift** at each step: (f) - (d) = "3D context contribution"

**Key question answered:** Does the model learn anything beyond "smaller
scaffolds have more sensitive positions"?

**Expected outcome:** Topology alone ~0.93-0.94; 3D context adds ~0.02-0.03.
Even a 0.02 lift is meaningful at this performance level.

**Script:** `scripts/ood/ablation_study.py`

---

### Experiment 2: Novel Scaffold Holdout
**Goal:** Test generalization to cores never seen during training.

**Protocol:**
1. Identify all unique core SMILES in the dataset
2. For each LOO-target fold:
   - Split test-target positions into "seen cores" (core appears in ≥1 training
     target) and "unseen cores" (core appears ONLY in test target)
   - Report NDCG@3 separately for seen vs unseen cores
3. Also: aggressive holdout — randomly hold out 20% of cores across ALL targets
   (scaffold-level split, not target-level)

**Key question answered:** Does performance degrade for truly novel scaffolds?

**Expected outcome:** Moderate degradation for unseen cores (features still
generalize, but less data to calibrate). This honestly shows the boundary.

**Script:** `scripts/ood/novel_scaffold_holdout.py`

---

### Experiment 3: Novel Target Family Holdout
**Goal:** Test generalization to protein families not in training.

**Protocol:**
1. Classify the 50 targets into families:
   - Kinases (~22 targets)
   - Epigenetic (BRD4, HDACs, KDM1A — ~4 targets)
   - Ion channels (KCNH2, Nav1.9 — ~2-3 targets)
   - Proteases (beta-secretase, DPP4 — ~2-3 targets)
   - GPCRs/receptors (orexin, P2X — ~3-4 targets)
   - Metabolic enzymes (MAO-B, AChE, CYP3A4 — ~3 targets)
   - Nuclear receptors (ER — 1 target)
   - Other (~8-10 targets)
2. Leave-one-family-out evaluation:
   - Train on all targets except family F
   - Test on family F
   - Report NDCG@3, Hit@1, Spearman per family
3. Special case: leave-ALL-kinases-out (22 targets removed from training,
   28 remain) — this is the hardest test since kinases are ~44% of the data

**Key question answered:** Is the model truly target-agnostic, or does it
require diversity of protein families in training?

**Expected outcome:** If the M7b finding (r>0.91 homogeneity) is real,
family holdout should show minimal degradation. If kinases secretly drive
performance, leaving them out will hurt.

**Script:** `scripts/ood/target_family_holdout.py`

---

### Experiment 4: Temporal Split
**Goal:** Gold-standard prospective validation — train on older data, test on
newer data.

**Protocol:**
1. Join MMP data with ChEMBL `docs` table to get `document_year` per assay
2. Define cutoff (e.g., 2018 or 2020 — whichever gives ≥20% test data)
3. Train on all MMPs with document_year ≤ cutoff
4. Test on MMPs with document_year > cutoff
5. Re-aggregate to position-level for both train and test
6. Report NDCG@3 and compare to LOO-target result

**Key question answered:** Does the model generalize to future chemistry, or
is it overfitting to historical compound distributions?

**Required data:** Need to join with ChEMBL SQLite `docs.year` column.

**Expected outcome:** Slight degradation expected (newer chemistry explores
different chemical space). A large drop would indicate the model captures
distributional artifacts rather than chemistry.

**Script:** `scripts/ood/temporal_split.py`

---

### Experiment 5: Feature Robustness / Sensitivity Analysis
**Goal:** Assess sensitivity to arbitrary choices in feature computation.

**Protocol:**
Sub-experiments:
- (a) **Pharmacophore radius**: Recompute context features at 3Å, 3.5Å, 4.5Å,
  5Å, 6Å. Retrain + evaluate.
- (b) **Conformer ensemble**: For 1,000 randomly sampled cores, generate 10
  ETKDG conformers each. Compute features on each. Measure coefficient of
  variation (CV) per feature. Report: what fraction of position rankings change
  when using a different conformer?
- (c) **Charge method**: For 1,000 cores, compare Gasteiger vs AM1-BCC charges.
  Report correlation and whether model performance changes.

**Key question answered:** How sensitive are predictions to implementation choices?

**Expected outcome:** Moderate CV for count features (±1-2 atoms at boundary);
low CV for charge/SASA. If rankings are stable across conformers, the model is
robust to conformational uncertainty.

**Script:** `scripts/ood/feature_sensitivity.py`

---

### Experiment 6: Low-Data Regime & Target Size Effects
**Goal:** Characterize performance as training data decreases.

**Protocol:**
1. Learning curve: subsample training data to 10%, 25%, 50%, 75%, 100%
   (stratified by target). Report NDCG@3 at each level.
2. Per-target analysis: correlate each target's test NDCG@3 with its
   number of training positions. Is there a minimum data threshold?
3. Simulate a "rare target" scenario: artificially limit each training target
   to 50, 100, 200, 500 molecules.

**Key question answered:** How much data does this approach need? Is it viable
for targets with <100 compounds?

**Expected outcome:** Graceful degradation. Performance likely plateaus around
25-50% of current data (the model learns general chemistry rules, not
target-specific patterns).

**Script:** `scripts/ood/learning_curve.py`

---

### Experiment 7: External Validation Against Published SAR Studies
**Goal:** Validate against expert-curated SAR knowledge from the literature.

**Protocol:**
1. **Bajorath activity cliff datasets** — Stumpfe & Bajorath have published
   curated cliff pair datasets. Compare position predictions to their annotations.
2. **Topliss tree validation** — Take the original Topliss substitution series
   (4-Cl, 3,4-diCl, 4-Me, 4-OMe, 4-H) on known scaffolds. Does the model
   correctly rank the para position as most SAR-sensitive?
3. **Published medchem case studies** — Select 3-5 published lead optimization
   campaigns (e.g., from J. Med. Chem.) where SAR at specific positions was
   systematically explored. Compare model rankings to reported SAR sensitivity.
4. **Selectivity cliff analysis** — For targets where selectivity is well-studied
   (e.g., COX-1 vs COX-2, JAK1 vs JAK2), check whether model correctly
   identifies the selectivity-determining positions.

**Key question answered:** Does the model agree with expert medchem knowledge?

**Expected outcome:** Qualitative agreement on major SAR features. Disagreements
are interesting — either the model is wrong, or it's finding patterns experts
haven't articulated.

**No script — manual curation required.** Document as a table of case studies
with model predictions vs literature SAR.

---

### Experiment 8: Directional Analysis (Sensitivity ≠ Actionability)
**Goal:** Characterize what "sensitive" means at each position — does it mean
"improvable" or "fragile"?

**Protocol:**
1. For high-sensitivity positions (top quartile), decompose the MMP distribution:
   - What fraction of modifications IMPROVE potency by >1 log unit?
   - What fraction DECREASE potency by >1 log unit?
   - What is the ratio of improvement vs degradation?
2. Classify positions as:
   - **Opportunity positions**: high sensitivity + more improvements than degradations
   - **Fragile positions**: high sensitivity + more degradations than improvements
   - **Neutral**: high sensitivity + balanced
3. Report the distribution of position types across the dataset
4. Test whether any features distinguish opportunity vs fragile positions

**Key question answered:** When the model says "sensitive," does that mean
"interesting to modify" or "don't touch this"?

**Expected outcome:** Most high-sensitivity positions are roughly balanced (both
improvements and degradations possible). This is actually the correct behavior
for a Topliss-type tool — you want to explore positions where potency *varies*,
regardless of direction.

**Script:** `scripts/ood/directional_analysis.py`

---

### Experiment 9: Statistical Significance — Permutation Tests
**Goal:** Establish null distribution for all reported metrics.

**Protocol:**
1. **Position model null:** Shuffle sensitivity labels within each (mol, target)
   group 1,000 times. Re-evaluate NDCG@3 each time. Report p-value for
   observed 0.964 vs null distribution.
2. **Change-type model null:** Shuffle |ΔpActivity| labels within each
   (core, target) group 1,000 times. Re-evaluate LOO Spearman. Report p-value
   for observed 0.268.
3. **Per-target significance:** For each of the 50 targets, report whether
   its individual Spearman is significant at p<0.05 after Bonferroni correction.
4. **Effect size:** Report Cohen's d for the improvement over random baseline.

**Key question answered:** Is the model capturing real signal, or could these
results arise by chance?

**Expected outcome:** NDCG@3 = 0.964 will be astronomically significant
(random ~0.85 for this task). Spearman 0.268 for change-type will be significant
but with p-values that vary by target.

**Script:** `scripts/ood/permutation_tests.py`

---

### Experiment 10: Calibration & Uncertainty Quantification
**Goal:** Add confidence estimates to predictions.

**Protocol:**
1. **Calibration curve:** Bin predicted sensitivities into deciles. Plot
   predicted vs actual mean sensitivity per bin. Report calibration error (ECE).
2. **Quantile regression:** Train HGB with `loss='quantile'` at α=0.1 and α=0.9
   to get 80% prediction intervals. Report interval widths and coverage.
3. **Conformal prediction:** Split-conformal or cross-conformal on LOO-target
   residuals. Report empirical coverage at 80% and 90% levels.
4. **Novelty detection:** Train an isolation forest or LOF on training features.
   Flag test positions that fall outside the training distribution. Report
   NDCG@3 separately for in-distribution vs out-of-distribution positions.

**Key question answered:** Can we tell the user "how confident" we are?

**Expected outcome:** Model is well-calibrated in the middle range but may
overpredict at extremes. Conformal intervals will be wide for novel scaffolds.

**Script:** `scripts/ood/calibration_uncertainty.py`

---

### Experiment 11: Hyperparameter Sensitivity
**Goal:** Demonstrate robustness to hyperparameter choices.

**Protocol:**
1. Random search (50 configurations):
   - max_iter: [100, 200, 300, 500, 1000]
   - max_depth: [3, 4, 5, 6, 7, 8, 10]
   - learning_rate: [0.01, 0.03, 0.05, 0.1, 0.2]
   - min_samples_leaf: [10, 20, 30, 50, 100, 200]
2. Report distribution of LOO-target NDCG@3 across configurations
3. Report best, worst, median, std
4. Show that performance is robust (low variance across configs)

**Key question answered:** Is the reported 0.964 sensitive to hyperparameter
choices, or does the result hold broadly?

**Expected outcome:** NDCG@3 range ~0.95-0.97 across reasonable configurations.
The conclusion is robust to tuning.

**Script:** `scripts/ood/hyperparam_sensitivity.py`

---

## Part 3: Priority & Ordering

### Must-have (run before public release)

| # | Experiment | Addresses | Effort |
|---|---|---|---|
| 1 | Feature ablation | A2, C1 (core_n_heavy critique) | Low (1h) |
| 9 | Permutation tests | A6, C1 (statistical significance) | Low (1-2h) |
| 2 | Novel scaffold holdout | A4, C5 (memorization) | Medium (2-3h) |
| 4 | Temporal split | C4 (gold-standard validation) | Medium (3-4h, needs ChEMBL join) |
| 8 | Directional analysis | A1 (sensitivity ≠ actionability) | Low (1-2h) |

### Should-have (strengthens the paper significantly)

| # | Experiment | Addresses | Effort |
|---|---|---|---|
| 3 | Target family holdout | C3, C7 (family generalization) | Medium (2-3h) |
| 10 | Calibration + uncertainty | C2 (confidence intervals) | Medium (3-4h) |
| 6 | Learning curve | C3 (low-data regime) | Low (1-2h) |
| 7 | External validation (manual) | A4 (independent validation) | High (half-day curation) |

### Nice-to-have (for thoroughness)

| # | Experiment | Addresses | Effort |
|---|---|---|---|
| 5 | Feature sensitivity | B1, B2, B3 (robustness) | Medium (3-4h) |
| 11 | Hyperparameter sensitivity | C6 (tuning robustness) | Low (1-2h) |

---

## Part 4: Honest Framing for Public Release

### What this model IS:
- A **triage tool** that ranks positions on a molecule by SAR sensitivity
- Trained on 25M matched molecular pairs from 50 diverse ChEMBL targets
- **Target-agnostic**: position sensitivity generalizes across protein families
- Uses only **ligand-side 3D features** — no protein structure required
- Change-type recommendations are **weakly predictive** (Spearman ~0.27) but
  consistent across targets and never anti-correlated

### What this model is NOT:
- Not a potency predictor (predicts *magnitude* of change, not *direction*)
- Not a substitute for protein-ligand modeling (no binding pocket features)
- Not validated prospectively (retrospective ChEMBL analysis only)
- Not suitable for absolute quantitative predictions (rankings are reliable;
  magnitudes are approximate)
- Not designed for multi-site cooperative SAR (single-cut MMPs only)

### Key limitations to acknowledge:
1. `core_n_heavy` accounts for ~45% of prediction — the "R-group is a bigger
   fraction of small molecules" effect is real and dominant
2. 3D features depend on a single computed conformer, not the bioactive one
3. Change-type model (Spearman 0.27) is useful for ranking but leaves 93% of
   transformation-level variance unexplained
4. All 50 training targets are data-rich (>4,800 activities each) — performance
   on rare targets is unknown
5. Training data is retrospective ChEMBL — no temporal validation yet

### Key strengths to highlight:
1. **Position-level reframe is the core insight** — transforms 3D context from
   useless (NDCG 0.41 at transformation level) to powerful (NDCG 0.96 at
   position level)
2. **Leave-one-target-out validation** is rigorous (50 folds, std 0.007)
3. **Target-agnostic generalization** is a genuine finding (r>0.91 across
   diverse protein families) — the rules of position sensitivity are universal
4. **Fully interpretable** — 11 physically meaningful features, SHAP attributions,
   real MMP evidence from ChEMBL
5. **No API keys, no protein structure, no special hardware** — runs from SMILES
   in seconds

---

## Part 5: Implementation Notes

### Directory structure for OOD experiments
```
scripts/ood/
├── ablation_study.py          # Exp 1
├── novel_scaffold_holdout.py  # Exp 2
├── target_family_holdout.py   # Exp 3
├── temporal_split.py          # Exp 4
├── feature_sensitivity.py     # Exp 5
├── learning_curve.py          # Exp 6
├── directional_analysis.py    # Exp 8
├── permutation_tests.py       # Exp 9
├── calibration_uncertainty.py # Exp 10
└── hyperparam_sensitivity.py  # Exp 11
```

Experiment 7 (external validation) is a manual curation task — results documented
in `outputs/ood/external_validation.md`.

All experiment outputs → `outputs/ood/` with JSON metrics + PNG plots.

### Data dependencies
- `evolve/eval_data/position_data.npz` — position-level features + labels
- `outputs/mmps/all_mmps.parquet` — full MMP corpus (for Exp 4, 8)
- `D:\Mike project data\...\chembl_36.db` — ChEMBL SQLite (for Exp 4 temporal join)
- `outputs/features/context_3d.parquet` — 3D context lookup (for Exp 5)

### Estimated total effort
- Must-have experiments: ~8-12 hours of compute + coding
- Should-have experiments: ~8-12 hours additional
- Nice-to-have: ~5-6 hours additional
- Total: ~2-3 focused sessions
