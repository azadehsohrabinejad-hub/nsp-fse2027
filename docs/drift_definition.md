# NSP Semantic Drift Definition

## 1. Purpose

Semantic Drift is a dynamic divergence between observable repair progress
and latent semantic alignment.

The central NSP signature occurs when an LLM repair agent appears to make
functional progress while moving away from the intended behavioral contract.

## 2. Hidden-State Components

At repair round t:

z_t = [p_t, s_t, c_t, r_t]^T

where:

- p_t: Repair Progress
- s_t: Semantic Alignment
- c_t: Context Quality
- r_t: Repair-Policy Stability

Define:

Delta p_t = p_t - p_{t-1}

Delta s_t = s_t - s_{t-1}

## 3. Semantic Decay

The step-wise semantic-decay severity is:

D_semantic(t) = max(0, -Delta s_t)

This quantity measures semantic deterioration regardless of observed
functional progress.

## 4. Deceptive Drift

A deceptive-drift event occurs when repair progress increases while semantic
alignment decreases.

Binary event definition:

I_deceptive(t) =
1(Delta p_t > delta_p and Delta s_t < -delta_s)

Continuous severity definition:

D_deceptive(t) =
max(0, Delta p_t) * max(0, -Delta s_t)

The binary definition identifies occurrence. The continuous definition
measures severity.

## 5. Probabilistic Drift

Because p_t and s_t are latent variables, drift should also be represented
probabilistically:

P_drift(t) =
P(Delta p_t > delta_p,
  Delta s_t < -delta_s
  | y_1:t)

A drift alarm is triggered when:

P_drift(t) > alpha

where alpha is a confidence threshold selected on validation data.

## 6. Trajectory-Level Severity

Cumulative drift burden:

D_traj =
sum_{t=2}^{T} D_deceptive(t)

Length-normalized severity:

D_traj_normalized =
(1 / (T - 1)) *
sum_{t=2}^{T} D_deceptive(t)

The first measures total drift burden. The second supports comparison across
trajectories of different lengths.

## 7. Sliding-Window Drift Detection

For a window of W transitions:

G_p(t,W) =
sum_{k=t-W+1}^{t} Delta p_k

G_s(t,W) =
sum_{k=t-W+1}^{t} Delta s_k

Windowed drift severity:

D_window(t) =
max(0, G_p(t,W)) *
max(0, -G_s(t,W))

A windowed alarm is triggered when:

G_p(t,W) > delta_p_window

and

G_s(t,W) < -delta_s_window

## 8. Drift Onset

Drift onset is defined as the earliest round at which the posterior
probability of drift exceeds the alarm threshold for m consecutive rounds:

t_onset =
min{t :
P_drift(k) > alpha
for all k in [t, t+m-1]}

The persistence requirement reduces isolated false alarms.

## 9. Innovation-Based Auxiliary Signal

For the Linear Gaussian baseline, the innovation is:

nu_t = y_t - C z_hat_{t|t-1}

with covariance S_t.

The normalized innovation statistic is:

J_t = nu_t^T S_t^{-1} nu_t

A large J_t indicates an unexpected observation but does not by itself prove
Semantic Drift.

Innovation statistics are used as auxiliary change-detection signals, while
the primary drift definition is based on the posterior changes in Repair
Progress and Semantic Alignment.

## 10. Mechanistic Attribution

Context Quality and Repair-Policy Stability are used to explain possible
drift mechanisms.

Context-associated deceptive drift:

D_context(t) =
D_deceptive(t) * max(0, -Delta c_t)

Instability-associated deceptive drift:

D_instability(t) =
D_deceptive(t) * max(0, -Delta r_t)

These are explanatory quantities and are not the primary definition of
Semantic Drift.

## 11. Difference from Static Test Overfitting

Static test overfitting is evaluated after a final patch is produced.

NSP Semantic Drift is a trajectory-level phenomenon. It identifies when the
repair process begins moving toward a test-passing but semantically degraded
state.

This enables inference-time intervention before the repair budget is
exhausted.

## 12. Validation Protocol

Round-level validation uses evidence unavailable to the repair agent during
inference:

- behavioral-contract satisfaction;
- held-out tests;
- ground-truth patch comparison;
- human semantic audit;
- per-round semantic-completeness labels.

Trajectory-level validation uses:

- final semantic completeness;
- test-passing-but-wrong classification;
- hidden-regression outcomes;
- final behavioral-contract satisfaction.

Evaluation metrics include:

- drift-event precision, recall, and F1;
- onset detection delay;
- false-alarm rate;
- lead time before semantic failure;
- trajectory-level drift-severity correlation with final semantic outcomes.