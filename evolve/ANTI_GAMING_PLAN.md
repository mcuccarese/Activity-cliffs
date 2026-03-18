# ShinkaEvolve Anti-Gaming & Overfitting Mitigation Plan

> **Purpose:** Identify ways the evolutionary process could game the fitness metric or overfit, and design concrete mitigations to ensure evolved scoring functions genuinely generalize.

---

## 1. Identified Risks

### Risk 1: Eval Data Memorization
**What:** The eval data is a fixed subsample (200 mol/target × 50 targets = ~1.19M rows, seed=42). Functions evolved over many generations against this same dataset can overfit to its specific statistical quirks.

**Why it matters:** A function that memorizes which feature combinations score high in this subsample won't generalize to new molecules, new targets, or different random samples.

**Mitigation:**
- **Multi-seed validation**: After evolution, re-evaluate the best functions on 5 different random subsamples (seeds 0-4). Report mean ± std NDCG@5. A function that drops >0.01 NDCG across seeds is overfitting.
- **Held-out target set**: Reserve 10 of the 50 targets as a pure validation set never seen during evolution. Report NDCG separately on the 40 "evolution" targets and 10 "holdout" targets.
- **Fresh data regeneration**: Periodically regenerate eval_data.npz with a new seed between evolution rounds.

### Risk 2: NDCG Metric Gaming via Score Distribution
**What:** NDCG only cares about relative ordering within each mol_from group. A function can get inflated NDCG by:
- Assigning extreme scores to items whose features happen to correlate with high |ΔpActivity| in this specific subsample
- Exploiting small group sizes (groups with 5-7 transforms are easy to rank by chance)

**Why it matters:** Small improvements in NDCG (~0.51 → ~0.52) could be noise, not signal.

**Mitigation:**
- **Statistical significance**: Bootstrap the NDCG calculation (resample mol_from groups with replacement, 1000 iterations). Report 95% confidence intervals. Only accept improvements outside the CI.
- **Minimum group size**: Already enforcing ≥5 transforms/group. Consider raising to ≥10 for validation.
- **Median NDCG**: Report both mean and median NDCG@5. Median is more robust to outlier groups.

### Risk 3: "Bigger Is Better" Degenerate Strategy
**What:** The simplest scoring function is |delta_HAC| + |delta_MW| — "bigger structural changes cause bigger activity changes." This gets NDCG@5 ≈ 0.51, which is already 17% above random (0.439). The useful signal is in the RESIDUAL: which SPECIFIC changes (not just bigger ones) are most informative.

**Why it matters:** An evolved function that merely learns "bigger = better" more precisely is trivially correct but scientifically useless for a Topliss tree. We need functions that distinguish WHICH modifications to prioritise, not just that larger modifications cause larger changes.

**Mitigation:**
- **Baseline-adjusted NDCG**: Compute NDCG on the residuals after regressing out |delta_HAC| and fp_tanimoto. This measures whether the function adds information beyond "bigger changes are bigger."
- **Controlled ablation**: For each evolved function, compute "NDCG improvement over |delta_HAC| alone" per target. Plot the distribution. Functions that only improve on easy targets are gaming.
- **Scientific interpretability gate**: Before accepting an evolved function, manually inspect it. Does it use specific feature interactions that map to pharmacological principles? Or is it just a complex rewrite of "bigger is better"?

### Risk 4: Feature Correlation Exploitation
**What:** Features are correlated (delta_MW and delta_HAC have r ≈ 0.95; delta_TPSA and delta_HBA are correlated). An evolved function can exploit collinearity to fit noise.

**Why it matters:** Learned coefficients on collinear features are unstable — they change drastically across subsamples.

**Mitigation:**
- **Feature stability test**: Evaluate the best function on 5 different subsamples. If the contribution of any single feature changes sign or magnitude by >50%, that feature's contribution is noise.
- **Simplified feature sets**: Also evolve on a reduced feature set (drop delta_MW as redundant with delta_HAC).

### Risk 5: Complexity Creep
**What:** ShinkaEvolve can generate arbitrarily complex functions with many branches, thresholds, and np.where statements. More complex functions fit better but generalise worse.

**Why it matters:** The goal is INTERPRETABLE rules (Topliss tree analog). A 50-line function with 15 magic thresholds is not interpretable.

**Mitigation:**
- **Complexity penalty**: Track AST node count for each candidate. Penalise functions with >30 AST nodes (subtract 0.001 NDCG per node above 30 from the fitness score).
- **Parsimony pressure**: In the ShinkaEvolve task prompt, explicitly instruct the LLM to prefer shorter, more interpretable functions.
- **Manual review gate**: Before deploying any evolved function, ensure a medchem expert can explain WHAT it does in 2-3 sentences.

### Risk 6: Within-Group Feature Collapse
**What:** All transforms from the same mol_from share similar attachment environment features (env_hash values). Within a group, env features have near-zero variance, so they contribute noise to within-group ranking.

**Why it matters:** A function using env features "looks" like it uses context, but the signal is entirely between-group (which molecules are in sensitive positions), not within-group (which transform to pick).

**Mitigation:**
- Already identified in M5b. Going forward:
- **Compute within-group feature variance** for every feature. Only evolve on features with meaningful within-group variance.
- **Separate between-group and within-group models**: Use env features to predict "is this molecule worth exploring?" (between-group) and transform features to predict "which modification?" (within-group).

### Risk 7: Symmetric MMP Leakage
**What:** Every MMP (A→B, delta=+2) has a twin (B→A, delta=-2). If both appear in eval data, predicting one leaks information about the other.

**Why it matters:** For leave-one-target-out (which we use), both twins are in the same target slice, so this doesn't cause train/test leakage. But if we ever do within-target cross-validation, it would be a problem.

**Mitigation:**
- Current leave-one-target-out design is safe.
- If within-target splits are added later, ensure symmetric pairs stay in the same fold.
- The current eval computes |abs_delta_pActivity| as the target, which is symmetric — this is correct.

### Risk 8: Target-Class Confounding
**What:** Feature distributions vary systematically across target classes (kinases vs proteases vs GPCRs). A function could learn "this feature pattern correlates with kinases, and kinases have higher cliffs on average" without learning anything causal about which modifications are informative.

**Why it matters:** This is a subtler form of the "bigger is better" problem — the function exploits base-rate differences across target classes rather than learning pharmacological rules.

**Mitigation:**
- **Per-target NDCG analysis**: Already computed. Check whether NDCG is uniformly improved across all target classes, or only for specific target families.
- **Target-class-balanced evaluation**: Weight NDCG by target class (kinase, protease, GPCR, etc.) to prevent kinase-heavy data from dominating.
- **Target-agnostic features**: The scoring function already doesn't receive target identity. This is good. But target-correlated feature distributions can still leak class information.

---

## 2. Implementation Plan

### Phase A: Hardened Evaluation (before next evolution round)
1. **Split targets**: 40 evolution / 10 holdout (stratified by target class)
2. **Add bootstrap CI** to NDCG calculation in manual_evolve.py
3. **Add baseline-adjusted NDCG** (residual after |delta_HAC| regression)
4. **Add median NDCG** alongside mean
5. **Add within-group feature variance** analysis

### Phase B: Feature Enrichment (richer features that have within-group variance)
1. **R-group FP difference bits** (256-dim XOR of Morgan FPs) — these DO vary within groups because different transforms change different bits
2. **R-group SMARTS categories** (adds_halogen, adds_amine, adds_aromatic, etc.) — high within-group variance
3. **Transform frequency** (how common is this specific R→R' swap across ChEMBL?) — varies per transform
4. **Within-group rank features** (how does this transform's |delta_HAC| compare to the group mean?) — by definition has within-group variance

### Phase C: Evolution with Guardrails
1. Add complexity penalty to fitness (AST nodes)
2. Add parsimony instructions to ShinkaEvolve task prompt
3. Run evolution on 40-target train set
4. Validate on 10-target holdout + multi-seed
5. Accept only if: holdout NDCG ≥ train NDCG - 0.01 AND bootstrap CI doesn't overlap baseline

---

## 3. Success Criteria

A scoring function is accepted as "generalizable" if ALL of the following hold:
1. **Holdout NDCG@5 ≥ 0.53** on the 10 unseen targets
2. **Multi-seed stability**: std < 0.005 across 5 random subsamples
3. **Baseline-adjusted improvement**: NDCG on residuals (after removing |delta_HAC|) > 0.46 (above random)
4. **Interpretability**: A medicinal chemist can explain the function's logic in ≤3 sentences
5. **Per-target consistency**: NDCG > 0.48 on ≥80% of individual targets (no catastrophic failures)
6. **Complexity**: ≤30 AST nodes in the scoring function body
