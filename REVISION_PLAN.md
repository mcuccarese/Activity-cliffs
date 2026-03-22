# Revision Plan: Addressing Reviewer Feedback

> **Purpose:** Concrete action items to address all three reviewer perspectives.
> Organized into phases by priority. Each item tagged with the reviewer
> critique it addresses (R1=Adopter, R2=Context, R3=Rejection).

---

## Identity Decision (Must Resolve First)

**R3's core demand is correct.** This is not an ML paper. The ML model is outperformed
by a heuristic. The paper's identity must be:

> **A large-scale chemoinformatics analysis that validates the universality of a
> simple chemical principle (R-group relative contribution scales inversely with
> scaffold size), delivers a practical MMP-backed triage tool, and honestly
> reports where ML fails to add value.**

The three genuine contributions are:
1. The **position-level reframe** as an insight
2. The **universality validation** across 50 targets, 6 families, temporal splits
3. The **25M MMP corpus + evidence-backed webapp** as a practical resource

The change-type model (Spearman 0.27) is a secondary contribution — honest, modest,
but the only component where ML adds value over a rule.

---

## Phase 1: Statistical Rigor (R3 Issues 2, 5, 6)

**Goal:** Every comparison in the paper has proper statistical backing.

### 1A. Paired statistical tests for heuristic vs HGB
**Script:** `scripts/ood/paired_tests.py`

For every experiment where "heuristic beats HGB" is claimed, run:
- **Wilcoxon signed-rank test** on per-target NDCG@3 pairs (50 paired observations)
- **Paired bootstrap CI** (10,000 resamples) on the NDCG@3 difference
- Report: delta, 95% CI, p-value, effect size (Cohen's d)

Data source: Per-target NDCGs already computed in each OOD experiment JSON.
Re-run ablation to capture all per-target breakdowns if not already saved.

Expected outcome: The differences are small (~0.007) but consistent across 50 targets.
Wilcoxon will likely show p < 0.05 given the consistency, but effect size will be tiny.
**This is the honest answer:** heuristic is statistically significantly better by a
trivially small margin.

### 1B. NDCG@3 operating range context
**Additions to results summary:**

Report a table showing:
| Metric | Random | Global mean | -core_n_heavy | HGB (11) |
|--------|--------|-------------|---------------|----------|
| NDCG@3 | 0.865 | 0.886 | 0.966 | 0.959 |
| Lift over random | — | +0.021 | **+0.101** | +0.094 |
| % of theoretical max | 0% | 16% | **75%** | 70% |

The "theoretical max" is 1.0 - 0.865 = 0.135 available headroom. The heuristic
captures 75% of available headroom. This is the honest framing: 0.966 sounds
amazing, but 57% of it is free (random baseline).

Also report: **median positions per molecule** (from position_data.npz groups).
If the typical molecule has 4-6 positions, NDCG@3 is ranking almost all of them,
making high scores easier.

### 1C. 3D features — formal test of incremental value
**Hypothesis test:** Do 3D features significantly improve Spearman over topology-only?

- Paired Wilcoxon on per-target Spearman: topology-only (2 feat) vs full (11 feat)
- Bootstrap CI on the Spearman difference
- If p > 0.05: 3D features are a negative result (no significant improvement)
- If p < 0.05: 3D features add marginal but significant Spearman improvement
  (+0.02), which is useful for cross-molecule calibration but not within-molecule ranking

Either way, the conclusion is honest. Present 3D features as an informative
negative for ranking, possible marginal positive for calibration.

**Effort:** 1 session. All data exists; this is analysis + reporting.

---

## Phase 2: Change-Type Model Hardening (R3 Issues 3, R1 concern, R2 context)

**Goal:** Give the change-type model (Spearman 0.27) the statistical scrutiny
it deserves, fix the correlated-axes problem, and add confidence indicators.

### 2A. Permutation test for change-type model
**Script:** `scripts/ood/change_type_permutation.py`

Protocol:
1. Shuffle |ΔpActivity| labels within each (core, target) group 1,000 times
2. Re-evaluate LOO Spearman for each permutation
3. Report: observed 0.268 vs null distribution, p-value, Cohen's d
4. Per-target: report which of the 50 targets have individually significant
   Spearman (Bonferroni-corrected p < 0.001)

Data: Re-use the 5M training sample from `train_change_type_model.py`.
The per-target Spearman values already exist in `change_type_meta.json`.

### 2B. Per-target performance analysis
Already available in `change_type_meta.json`. Present as:
- Histogram of per-target Spearman (range 0.14–0.42)
- Bottom quartile vs top quartile: is there a pattern? (target family, data size, cliff rate)
- Report: "X/50 targets above 0.3, Y/50 above 0.2, Z/50 above 0.15, all > 0"

### 2C. Address correlated axes problem
**This is the most substantive change-type revision.**

Current issue: the 11 delta-prop axes are probed independently, but they're correlated.
EWG and lipophilicity co-occur (Cl, CF3). The recommendation "try EWG" vs "try
lipophilic" may be saying the same thing twice.

**Fix — Option A (simpler, preferred):** Orthogonalized probing
1. Compute correlation matrix of the 11 delta-prop columns across training data
2. PCA → 11 orthogonal components
3. Probe along principal components instead of raw axes
4. Label components by their dominant loading ("electronic character," "size/lipophilicity," etc.)
5. Report: how many PCs explain 90% of variance? (probably 5-6)

**Fix — Option B (simpler still):** Group correlated axes
- Manually group: {EWG + EDG + charge} = "electronic", {lipophilicity + heavy atoms + fsp3} = "size/lipophilicity", {rings + aromatic rings} = "ring system", {HBD + HBA} = "polarity"
- Report per-group scores instead of per-axis
- This is more interpretable for chemists

**Recommendation:** Implement Option B for the webapp (chemists understand grouped
categories), report Option A in the paper (statistical rigor), show that the
top recommendations are robust to orthogonalization.

### 2D. Confidence indicator in webapp
**File:** `webapp/app.py`, `webapp/predict.py`

For each change-type recommendation, show a confidence tier:
- **Strong signal** (cliff_score > y_mean + 1σ): The probed axis predicts notably
  larger cliffs than average at this context
- **Moderate signal** (cliff_score > y_mean): Above-average cliff prediction
- **Weak signal** (cliff_score ≤ y_mean): At or below average — low confidence

Also add a global disclaimer: "Change-type rankings explain ~7% of transformation-level
variance (LOO Spearman 0.27). Use as a triage tool, not a design tool."

**Effort:** 2 sessions. Permutation test + correlated axes fix + webapp UI.

---

## Phase 3: External Validation (R3 Issue 4, R1 concern, R2 context)

**Goal:** Validate against published SAR campaigns where ground truth exists
independently of ChEMBL training data.

### 3A. Select 5 published case studies
**Criteria:**
- Published in J. Med. Chem. or similar, with systematic position-by-position SAR
- Scaffold NOT in training data (temporal or chemical novelty)
- Author explicitly identifies which positions are SAR-sensitive

**Candidate campaigns:**
1. **EGFR T790M mutant inhibitors (post-2018):** New scaffolds designed around
   osimertinib. Positions driving selectivity over wild-type are well-documented.
2. **PROTAC linker optimization:** Novel scaffold class (post-2020), SAR driven
   by linker attachment point and exit vector — fundamentally position-level.
3. **KRAS G12C covalent inhibitors:** Entirely new class (sotorasib/adagrasib era).
   The position sensitivity pattern (warhead > solvent-exit > pocket-filling) is
   published and testable.
4. **JAK1 vs JAK2 selectivity:** Positions determining isoform selectivity are
   textbook knowledge. Can the model identify them?
5. **BET bromodomain chemical probes:** JQ1 and I-BET series have exhaustive
   published SAR at each position.

### 3B. Validation protocol
For each case study:
1. Input the parent scaffold SMILES
2. Record model's position ranking (before looking at published SAR)
3. Compare to published SAR sensitivity (expert-annotated from the paper)
4. Score: does the model's top-1 (or top-2) position match the known SAR hotspot?
5. Record hit/miss and any interesting disagreements

### 3C. Topliss tree validation (R1 specific request)
Take 3 classic Topliss substitution series:
- 4-chloro phenyl ring (the original 1972 example)
- Benzoic acid derivatives
- Pyridine ring substitution

Run the position model. Does it correctly rank the para position (the position
Topliss's tree was designed for) as SAR-sensitive?

**Effort:** 1–2 sessions. Mostly manual literature curation + running the model.
No new code beyond a simple validation script that runs predict_positions on
a list of SMILES and saves results.

---

## Phase 4: Webapp Improvements (R1 concerns, R3 Issue 2)

### 4A. Honest framing in the UI
Add a collapsible "About this tool" section to the webapp:

```
## How it works
Position sensitivity is predicted primarily by scaffold size: positions on smaller
scaffolds are more SAR-sensitive because the R-group contributes a larger fraction
of the binding interaction. This simple principle (NDCG@3 = 0.966) holds across all
50 ChEMBL targets tested, all protein families, and temporal splits.

The ML model adds marginal calibration via 3D pharmacophore context but does NOT
improve position ranking over the core-size heuristic.

Change-type recommendations (which axis of modification to try) are weakly predictive
(Spearman 0.27, ~7% variance explained). Use them to narrow from 11 axes to 3–4, not
as definitive guidance.

## What this tool IS:
- A triage tool for prioritizing positions to explore
- Backed by 25M real experiments from ChEMBL
- Target-agnostic: works for any target without retraining

## What this tool is NOT:
- Not a potency predictor (magnitude, not direction)
- Not a substitute for protein-aware modeling
- Not validated prospectively
```

### 4B. Flexibility warning
In `predict.py`, compute number of rotatable bonds. If > 5, show a yellow warning:
"This molecule has {n} rotatable bonds. 3D features are computed from a single
conformer and may be less reliable for flexible molecules. Position ranking by
core size remains valid."

### 4C. Baseline context display
Show a small info box: "Random baseline: 0.865 NDCG@3. This model: 0.966.
Lift: +0.101 on a 0.135 headroom scale."

**Effort:** 1 session. UI changes only.

---

## Phase 5: Paper Framing & Literature (R2 entire review)

### 5A. Literature to cite and contextualize

**Topliss tree lineage:**
- Topliss JG, J. Med. Chem. 1972, 15, 1006 — original tree
- Craig PN, J. Med. Chem. 1971 — independent substituent mapping
- Topliss JG & Martin YC, J. Med. Chem. 1975 — multivariate extension
- Leach et al. (FBDD-era updates) — modern extensions

**MMP and activity cliff literature:**
- Hussain J & Rea C, JCICS 2010 — MMP algorithm (rdMMPA basis)
- Stumpfe D & Bajorath J, JCICS 2012, JMC 2014, multiple reviews through 2024 —
  activity cliff definition, curated datasets
- Kramer C et al., JCICS 2014 — MMP significance and experimental uncertainty
- Tyrchan C & Evertsson E, JCIM 2017 — MMP limitations review
- Dossetter AG et al., MedChemComm 2013 — large-scale MMP analysis (AZ)
- Warner DJ et al., JCIM 2019 — MMP property prediction

**Position-level SAR / landscape analysis:**
- Guha R & Van Drie JH, JCICS 2008 — SALI (Structure-Activity Landscape Index)
- Naveja JJ & Medina-Franco JL, Future Med Chem 2015; JCIM 2019 — activity landscape
- Schneider et al. (2010–2020) — scaffold-level SAR profiling

**Ligand efficiency theory (core_n_heavy context):**
- Hopkins AL et al., Drug Discov Today 2004 — ligand efficiency
- Congreve M et al., Drug Discov Today 2003 — Rule of Three
- Hajduk PJ & Greer J, Nat Rev Drug Discov 2007 — fragment theory
- Murray CW & Rees DC — fragment contributions to binding

**ML for activity cliff prediction:**
- van Tilborg D et al., JCIM 2022 — GNN cliff prediction
- Deng J et al., Briefings in Bioinformatics 2023 — deep learning cliffs
- Iqbal J et al., JCIM 2021 — ML MMP potency prediction
- van Westen GJP et al., JCIM 2013 — proteochemometric modeling (target similarity)

### 5B. Narrative structure for the paper

**Introduction framing:**
- Topliss (1972) encoded expert knowledge about σ/π parameters as a decision tree
- 50 years of SAR data now in ChEMBL (25M MMPs from 50 targets)
- Can we learn the equivalent rules from data?
- Answer: partially. We validate that a simple physical principle (R-group relative
  contribution) dominates position sensitivity universally, and provide weak but
  genuine signal for which modification type to try

**Results structure:**
1. Position-level reframe (transformation NDCG 0.52 → position NDCG 0.97)
   - Present as the key intellectual insight
2. core_n_heavy dominance (ablation, all baselines)
   - Present as validation of ligand efficiency theory at scale
   - **Not a bug, a finding**
3. Universality (target families, temporal, novel scaffolds)
   - The genuinely novel empirical contribution
4. What ML doesn't add (3D features negative result)
   - Present honestly as informative negative
5. What ML does add (change-type Spearman 0.27)
   - Present modestly as triage-level signal
6. The webapp as practical contribution
   - Evidence-backed, not model-backed

### 5C. Negative results to discuss explicitly

1. **Interaction features failed (M6c):** The context × change_type cross-product
   hypothesis was theoretically motivated (provides within-group variance) but
   empirically null. Why? Because HGB already models interactions natively, and
   the pre-computed interactions are redundant. This should be discussed as a
   lesson for feature engineering in MMP analysis.

2. **Pharmacophore homology failed (M7b):** All 50 targets produce the same SAR
   patterns (r > 0.91). Either the features are too coarse to distinguish targets,
   or position sensitivity is genuinely target-agnostic. The OOD results support
   the latter — but acknowledge both interpretations.

3. **ShinkaEvolve lessons (M5):** 16 candidate scoring functions explored, none
   broke the ceiling. The search space of algebraic combinations of 2D descriptors
   is fundamentally limited. Discuss what this means for automated scoring function
   discovery.

**Effort:** 2–3 sessions of writing. No code.

---

## Phase 6: Remaining OOD Experiments (R3 Issues 6, 7)

### 6A. Experiment 5: Conformer sensitivity (optional but strengthening)
Test how much predictions change across 10 ETKDG conformers for 1,000 cores.
This directly addresses R2/B1 ("single conformer is not bioactive").

If prediction rankings are stable across conformers: the model is robust
to conformational noise (strong defense).
If rankings change: quantify the fraction of swapped rankings, flag as limitation.

### 6B. Experiment 10: Calibration + uncertainty (optional but valuable)
Conformal prediction on LOO residuals → 80% prediction intervals.
Add to webapp: "Predicted sensitivity: 1.2 ± 0.3 (80% CI)."

### 6C. Experiment 11: Hyperparameter sensitivity (trivial, just run it)
50-config random search. Show NDCG@3 varies <0.02 across configs.
Proves the result isn't fragile.

**Effort:** 2 sessions total for all three.

---

## Persistent Limitations (Cannot / Should Not Fix)

These are fundamental scope boundaries. The revision plan acknowledges them
explicitly rather than trying to engineer around them.

### L1. Single-cut MMPs miss cooperative SAR
**Why it persists:** Multi-cut MMPs are exponentially noisier (n² pairs for double-cut)
and the attribution of ΔpActivity to specific positions becomes ambiguous. Single-cut
is the methodological standard (Hussain & Rea, 2010).

**Advice for users:** The model assumes positions are independent. For lead
optimization where you're modifying 2 positions simultaneously, use the model
to prioritize which positions to explore, but validate cooperative effects
experimentally. The evidence panel may show multi-position analogs.

### L2. Sensitivity, not direction
**Why it persists:** Direction (improve vs degrade) depends on protein binding pocket
geometry, which no ligand-only feature captures. The directional analysis (Exp 8)
confirms this: direction and sensitivity are independent (rho = -0.005). Adding
direction would require protein structure or target-specific models.

**Advice for users:** "Sensitive" means "variable." High-sensitivity positions are
equally likely to improve or degrade potency (29%/29% split, Exp 8). This is a
feature, not a bug — the tool identifies where to explore, like the original Topliss
tree. Use the evidence panel to see what direction previous experiments took.

### L3. Gasteiger charges are crude
**Why it persists:** Gasteiger is the only charge method available in pure RDKit with
no external dependencies (AM1-BCC requires OpenBabel or ORCA). For a zero-install
Streamlit app, this is the pragmatic choice. Furthermore, the ablation shows 3D
features (including charge) don't improve ranking, so upgrading charges would not
change the headline result.

**Advice for users:** The model's rankings are driven by core size, not charge. The
Gasteiger charge contributes to absolute sensitivity calibration (Spearman +0.02).
For molecules with unusual charge distributions (zwitterions, highly charged species),
predictions may be less calibrated but ranking will still be correct.

### L4. Free-ligand 3D features ≠ bound-state
**Why it persists:** Bound-state features require protein crystal structures, which
defeats the purpose of a "SMILES-in, answer-out" tool. The model learns statistical
correlates of sensitivity from free-ligand features, not binding physics. This is
validated empirically: the model works despite using free-ligand features, because
the dominant signal (core size) is conformation-independent.

### L5. Target selection bias (50 data-rich targets)
**Why it persists:** The model requires many MMPs per target to learn position
sensitivity. Targets with <100 compounds don't generate enough MMPs for meaningful
aggregation to position level. This is a data limitation, not a methodological one.

**Advice for users:** The model has been validated on data-rich targets (>4,800
activities each). For novel or data-sparse targets, the core-size heuristic still
applies (it's target-agnostic and data-free), but the evidence panel will have fewer
supporting examples. The learning curve (Exp 6) shows the model works with as few
as 50 positions per target, suggesting it would transfer to moderately-studied targets.

### L6. Change-type model explains only 7% of variance
**Why it persists:** Transformation-level cliff prediction is fundamentally harder
than position-level sensitivity. Which specific R-group swap causes a cliff depends
on protein-R-group interactions that no ligand-only descriptor captures. The 0.27
Spearman is the ceiling for ligand-only transformation prediction — breaking it
requires protein features (docking scores, binding pocket descriptors, or protein-
ligand co-crystal data).

**Advice for users:** Use change-type rankings to narrow from 11 modification axes
to 3–4 "most likely to cause SAR movement." This is a Topliss-style triage: "start
with EWG, then lipophilic, then size" narrows the initial analog set. Do not use
the cliff_score values as quantitative predictions — they're ordinal, not interval.
The evidence panel provides the real quantitative data.

---

## Execution Order

| Phase | Sessions | Prerequisite | Blocks |
|-------|----------|-------------|--------|
| **1. Statistical rigor** | 1 | None | Phase 5 (need stats for paper) |
| **2. Change-type hardening** | 2 | None | Phase 4D (confidence tiers need thresholds) |
| **3. External validation** | 1–2 | None | Phase 5 (need results for paper) |
| **4. Webapp improvements** | 1 | Phase 2 | None |
| **5. Paper framing** | 2–3 | Phases 1–3 | None |
| **6. Remaining OOD** | 2 | None | None |

**Phases 1, 2, 3 can run in parallel** (independent). Phase 4 depends on Phase 2.
Phase 5 depends on 1–3 (needs all results).

**Minimum viable revision:** Phases 1 + 3 + 5 (statistical rigor + external validation +
honest framing). This addresses R3's core demands and R2's literature gaps.

**Full revision:** All 6 phases.

---

## Model Recommendation per Phase

| Phase | Model | Rationale |
|-------|-------|-----------|
| 1 (statistics) | Sonnet | Clear protocol, well-scoped code |
| 2A–B (permutation/analysis) | Sonnet | Mechanical computation |
| 2C (correlated axes) | Opus | Scientific judgment on grouping |
| 3 (external validation) | Opus | Requires medchem domain reasoning |
| 4 (webapp) | Sonnet | UI code changes |
| 5 (paper framing) | Opus | Scientific writing + domain synthesis |
| 6 (remaining OOD) | Sonnet | Adapting existing experiment scripts |
