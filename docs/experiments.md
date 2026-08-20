# NSP Paper Experimental Design

## Scope Note

This document separates the experimental plan into two layers:

1. **Paper 1 / Thread 1 — Core evaluation**  
   Formal NSP specification, identifiability, estimation, finite-sample behavior, and fit to CCBench trajectories.

2. **Thread 3 — Drift detection and early intervention extension**  
   Inference-time drift alarms, onset detection, prediction of test-passing-but-wrong outcomes, and intervention value.

The two layers should not be merged unless the target paper scope is intentionally expanded.

---

# Part A — Paper 1 / Thread 1 Core Experiments

## RQ1 — Can NSP recover latent dynamics under known ground truth?

### Objective
Evaluate whether the proposed NSP estimators recover model parameters and latent states on synthetic trajectories generated from known parameters.

### Methodology
- Generate synthetic trajectories under known values of transition, observation, and noise parameters.
- Fit gradient-based MLE, Classical EM, and Particle-filter MLE.
- Repeat across multiple seeds, trajectory lengths, and noise regimes.

### Metrics
- Parameter recovery error after latent-space alignment.
- Hidden-state RMSE.
- One-step predictive log-likelihood.
- Uncertainty-interval coverage.
- Convergence and numerical-failure rates.

## RQ2 — Does trajectory structure improve fit over simpler baselines?

### Objective
Determine whether NSP explains repair trajectories better than non-sequential or simpler sequential models.

### Baselines
- i.i.d. independent-feature model.
- Trajectory-length-only summary model.
- Discrete-state HMM.
- Linear Gaussian state-space baseline.
- Mixed-likelihood NSP when raw round-level data are available.

### Metrics
- Held-out per-trajectory log-likelihood.
- Average NLL per observation.
- AIC/BIC where comparable.
- Calibration and posterior predictive checks.
- Runtime and memory cost.

## RQ3 — How robust are NSP estimates across models, strategies, and data conditions?

### Objective
Evaluate stability and cross-model consistency of recovered dynamics.

### Methodology
- Fit separately by model, strategy, mutation family, and repository.
- Repeat across seeds and initializations.
- Evaluate sensitivity to state dimension, sequence length, missingness, normalization, and noise assumptions.

### Metrics
- Cross-model consistency.
- Bootstrap confidence intervals.
- Between-seed variability.
- Ablation stability.
- Held-out repository or mutation-family performance.

## RQ4 — What are the finite-sample limits of NSP estimation?

### Objective
Measure estimator quality under short APR trajectories.

### Methodology
- Simulate T = 2, 5, 10, 20, 50.
- Compare Classical EM, gradient MLE, and Particle-filter MLE.
- Vary noise and sample size.

### Metrics
- MSE, bias, variance.
- Gap to CRB where derivable.
- Convergence probability.
- Computational cost.
- Minimum stable trajectory length.

---

# Part B — Thread 3 Drift Detection Extension

## RQ-D1 — Can NSP detect semantic drift?

### Ground-truth labels
Round-level labels should come from evidence unavailable to the agent during inference, such as per-round semantic completeness, contract violations, held-out tests, ground-truth patch divergence, or independent human audit.

### Metrics
- Precision, recall, F1.
- False-alarm rate.
- Onset delay.
- Lead time.
- AUROC/AUPRC.

## RQ-D2 — Does NSP outperform heuristic drift detectors?

### Baselines
- Static test-pass threshold.
- Pass-rate change threshold.
- Context-limit threshold.
- Edit-instability threshold.
- Innovation-only detector.
- Logistic regression on raw observations.

### Metrics
Use the same event-level and trajectory-level metrics as RQ-D1, plus calibration and paired method comparisons.

## RQ-D3 — Can early NSP states predict test-passing-but-wrong outcomes?

### Methodology
- Extract posterior summaries at normalized early checkpoints.
- Compare classifiers using NSP states, NSP drift features, and raw-observation baselines.

### Metrics
- AUROC, AUPRC.
- Precision@K.
- Brier score and calibration.
- Repair-budget savings.

## RQ-D4 — What is the value of NSP-triggered intervention?

### Offline policy evaluation
Simulate stop, context reset, strategy switch, or verification at detected onset.

### Metrics
- Semantic success rate.
- Test-passing-but-wrong rate.
- Rounds and tokens saved.
- False-intervention cost.
- Net utility.

---

# Evaluation Protocol

## Data splitting
Use grouped splits to avoid leakage. Prefer repository holdout, then task holdout, then grouped cross-validation by task. A random split stratified only by model × strategy is insufficient when the same task appears in multiple runs.

## Cross-validation
- Use grouped 5-fold CV for tuning and robustness analysis.
- Keep one untouched final test split.
- Fit normalization only on training data.

## Statistical analysis
- Paired bootstrap for F1, AUROC, AUPRC, and lead-time differences.
- Permutation tests for paired method comparisons.
- Wilcoxon only when paired scalar outcomes and assumptions are suitable.
- Apply multiple-comparison correction where needed.
- Report effect sizes and 95% confidence intervals.

## Reproducibility
Record dataset version, split manifest, random seed, model version, feature definition, normalization parameters, estimator configuration, checkpoint hash, and code commit.

---

# Decision

For the current FSE Paper 1 roadmap, **Part A is the core experimental plan**.

Part B is valuable but belongs primarily to the later inference-time detection/intervention thread unless the paper scope is intentionally expanded and sufficient round-level semantic labels are available.
