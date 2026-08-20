 NSP Hidden-State Definition

 1. Purpose

The Neural State-Space Process represents each LLM repair trajectory as a
sequence of latent states.

At repair round t, the hidden state is defined as:

z_t = [p_t, s_t, c_t, r_t]^T

where:

- p_t represents Repair Progress;
- s_t represents Semantic Alignment;
- c_t represents Context Quality;
- r_t represents Repair-Policy Stability.

The hidden state is not directly observed. It is inferred from the
round-level observation vector y_t.

The four dimensions are intended as scientifically interpretable latent
constructs. Their semantic interpretation must be validated using external
criteria and cannot be established solely by naming the latent coordinates.

 2. State Domain

For the interpretable NSP representation, each latent component is mapped
to the interval [0,1]:

z_t in [0,1]^4

Interpretation:

- values near 1 indicate a desirable state;
- values near 0 indicate a degraded state.

The Linear Gaussian baseline may maintain an unconstrained internal state
in R^4. For reporting and interpretation, this state can be mapped to
[0,1]^4 using a logistic transformation.

 3. Repair Progress

 Definition

Repair Progress measures the latent degree to which the current repair has
advanced toward functional resolution of the task.

It is not identical to test pass rate. Instead, it is inferred from multiple
signals such as:

- passed and failing test counts;
- resolved failures;
- newly introduced failures;
- build status;
- changes in test performance across rounds.

 Interpretation

High p_t:
- failures are being resolved;
- functional behavior is improving;
- regressions are limited.

Low p_t:
- little functional progress has been made;
- failures persist or increase;
- the repair process is stalled.

 External validation

Repair Progress can be validated against:

- held-out test performance;
- final repair success;
- number of unresolved failures;
- time or rounds required to reach a valid repair.

 4. Semantic Alignment

 Definition

Semantic Alignment measures the latent consistency of the current patch with
the intended behavioral contract of the task.

This is the primary semantic axis of the NSP model.

High test performance does not necessarily imply high Semantic Alignment.

 Interpretation

High s_t:
- the patch preserves the intended behavioral contract;
- the repair addresses the actual defect;
- hidden invariants and edge cases are preserved.

Low s_t:
- the patch overfits visible tests;
- the repair is incomplete or behaviorally incorrect;
- the patch introduces a hidden semantic regression.

 External validation

Semantic Alignment must be validated using criteria not exposed to the
repair agent during inference, including:

- semantic completeness;
- behavioral-contract satisfaction;
- ground-truth patch comparison;
- independent human or LLM semantic audit;
- test-passing-but-wrong labels.

Ground-truth information is used only for post-hoc evaluation and must not
be included in the agent's runtime observation space.

 5. Context Quality

 Definition

Context Quality measures the latent relevance, sufficiency, and usability
of the information available to the repair agent at round t.

It is not equivalent to context size.

 Interpretation

High c_t:
- relevant source files and tests are available;
- the failure evidence is correctly localized;
- context truncation is limited;
- retrieved information supports the repair objective.

Low c_t:
- relevant files are missing;
- retrieved context is noisy, stale, or unrelated;
- important evidence is truncated;
- the agent focuses on incorrect locations.

 External validation

Context Quality can be validated against:

- target-file recall and precision;
- relevance judgments for retrieved files;
- behavioral-contract coverage;
- localization accuracy;
- ablation experiments removing or replacing retrieved context.

 6. Repair-Policy Stability

 Definition

Repair-Policy Stability measures whether the observable repair behavior is
coherent, non-cyclic, and progressively responsive to execution feedback.

The term refers to observable repair behavior rather than private
chain-of-thought reasoning.

 Interpretation

High r_t:
- edits respond consistently to test evidence;
- repeated failures lead to meaningful policy changes;
- the repair scope remains controlled;
- previously resolved issues are not repeatedly reintroduced.

Low r_t:
- the agent repeats ineffective edits;
- edits are reverted or oscillate across rounds;
- token use grows without measurable progress;
- patch scope expands unnecessarily;
- the agent repeatedly breaks previously working behavior.

 External validation

Repair-Policy Stability can be validated using:

- repeated-edit ratio;
- reverted-edit ratio;
- recurrence of identical failures;
- oscillation between patch states;
- invalid-patch frequency;
- build breakage and regression frequency;
- progress per token or per round.

 7. State Dynamics

The general transition model is:

z_t ~ p(z_t | z_{t-1}, u_t; theta)

where u_t may contain exogenous inputs such as:

- test feedback;
- context changes;
- strategy selection;
- model family;
- repair budget;
- reward or stopping signals.

The Linear Gaussian baseline uses:

z_t = A z_{t-1} + B u_t + w_t

where:

w_t ~ N(0,Q)

The matrix A represents the persistence and cross-coupling of the four
latent dimensions.

Examples of possible cross-effects include:

- improved Context Quality increasing later Repair Progress;
- decreasing Repair-Policy Stability reducing Semantic Alignment;
- increasing test-based Repair Progress while Semantic Alignment declines.

The last pattern is a central candidate signature of Semantic Drift.

 8. Observation Relationship

The observation model connects the hidden state to the measured repair
signals:

y_t ~ p(y_t | z_t; theta)

The baseline uses:

y_t = C z_t + v_t

where:

v_t ~ N(0,R)

The final mixed-likelihood model uses feature-specific observation
distributions.

The loading structure should be constrained or regularized so that each
latent dimension remains interpretable.

 9. Identifiability and Interpretability

Latent-state coordinates are not automatically identifiable by semantic
meaning.

Equivalent models may differ through:

- permutation of latent dimensions;
- sign inversion;
- scaling;
- linear rotation.

To improve interpretability, NSP v1.0 will use:

1. fixed state dimension K = 4;
2. anchor observations for each latent construct;
3. sign and scale conventions;
4. structured or sparse observation loadings;
5. external validation criteria;
6. synthetic recovery experiments;
7. sensitivity analysis under multiple initializations.

 10. Proposed Anchor Observations

 Repair Progress anchors

- pass rate;
- resolved failure count;
- failing test count;
- new failure count;
- build status.

 Semantic Alignment anchors

Runtime proxy signals:
- regression indicators;
- destructive deletion;
- scope expansion;
- contract-related signals when available.

Post-hoc validation signals:
- semantic completeness;
- ground-truth agreement;
- human audit labels.

 Context Quality anchors

- relevant-file retrieval;
- target-file coverage;
- context truncation;
- irrelevant-file ratio;
- retrieval recurrence.

 Repair-Policy Stability anchors

- repeated edit ratio;
- reverted edit ratio;
- repeated failure ratio;
- invalid patch indicator;
- oscillation and scope expansion.

 11. Research Interpretation

The NSP does not claim that the four latent dimensions are directly
observed psychological states of the LLM.

They are trajectory-level latent constructs that summarize observable repair
behavior.

Their interpretation is accepted only when supported by:

- loading patterns;
- synthetic recovery;
- external validity;
- predictive usefulness;
- stability across models, tasks, and strategies.