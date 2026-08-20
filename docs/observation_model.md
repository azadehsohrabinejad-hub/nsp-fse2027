 NSP Observation Model Definition

 1. Purpose

In the NSP framework, the observation vector y_t contains heterogeneous
round-level signals from an LLM-based repair trajectory.

The current Linear Gaussian baseline approximates the complete observation
vector as:

y_t | z_t ~ N(C z_t, R)

This approximation is retained as a computational baseline. However, the
features have different supports and data-generating mechanisms. The target
NSP observation model therefore uses feature-specific likelihoods.

Unless explicitly modeled otherwise, the first implementation assumes
conditional independence of observation channels given the latent state:

p(y_t | z_t) = product_j p(y_tj | z_t)

This assumption will be evaluated empirically using residual correlations
and posterior predictive checks.

 2. Test Features

 passed_tests / total_tests

Preferred representation when both counts are available:

passed_tests_t ~ Binomial(total_tests_t, p_t)

or, when additional dispersion is observed:

passed_tests_t ~ Beta-Binomial(total_tests_t, alpha_t, beta_t)

The probability p_t is linked to the latent state through a logistic link.

 pass_rate

The pass rate should be treated as a derived feature rather than an
independent observation whenever passed_tests and total_tests are available.

If only pass_rate is available:

- values strictly inside (0,1): Beta distribution;
- exact values at 0 or 1: zero-one-inflated Beta distribution.

 delta_pass_rate

Bounded continuous value in [-1,1].

Baseline approximation:
- truncated Gaussian.

Preferred final treatment:
- derive it from consecutive test-count observations rather than model it
  as an independent channel.

 failing_test_count
 new_failure_count
 resolved_failure_count

Non-negative count variables.

Candidate likelihoods:
- Poisson when variance is approximately equal to the mean;
- Negative Binomial under overdispersion;
- zero-inflated variants when excessive zeros are observed.

The final selection must be data-driven.

 3. Edit Features

 edit_count
 files_touched
 lines_added
 lines_removed

Non-negative count variables.

Candidate likelihoods:
- Poisson;
- Negative Binomial;
- zero-inflated Poisson or Negative Binomial.

Negative Binomial is the preferred initial model for edit_count,
lines_added, and lines_removed because repair activity is likely to be
overdispersed.

 repeated_file_edit_ratio
 reverted_edit_ratio

Bounded continuous values in [0,1].

Candidate likelihoods:
- Beta for values in (0,1);
- zero-one-inflated Beta when exact 0 and 1 values occur.

 4. Context Features

 retrieved_file_count

Non-negative count:
- Poisson or Negative Binomial;
- potentially zero-inflated.

 context_tokens

Non-negative count:
- Negative Binomial as the initial likelihood;
- Log-Normal may be considered when treated as an approximately continuous,
  heavy-tailed quantity.

 context_truncated

Binary observation:

context_truncated_t ~ Bernoulli(pi_t)

with pi_t linked to z_t using a logistic link.

 5. Usage Features

 input_tokens
 output_tokens

Non-negative, typically right-skewed variables.

Candidate likelihoods:
- Negative Binomial when modeled as counts;
- Log-Normal when treated as continuous heavy-tailed measurements.

The selection will be based on dispersion, zero frequency, and
posterior-predictive fit.

 latency_seconds

Strictly positive and typically right-skewed.

Candidate likelihoods:
- Log-Normal;
- Gamma.

Model selection will be performed empirically.

 6. Behavioral Features

The following variables are binary:

- invalid_patch
- build_broken
- test_regression
- scope_expansion

For each binary feature b_jt:

b_jt ~ Bernoulli(pi_jt)

where:

logit(pi_jt) = a_j + w_j^T z_t

 7. Missing Data

Missing values must not automatically be encoded as zero.

For every feature, the system records:

- observed value;
- missingness indicator;
- source provenance.

Missing channels are excluded from the round-level likelihood rather than
treated as observed zeros.

 8. Baseline and Target Models

 Baseline

The Linear Gaussian State-Space Model uses:

y_t | z_t ~ N(C z_t, R)

after feature normalization.

This model serves as a reproducible baseline and supports exact Kalman
filtering and classical EM.

 Target Mixed-Likelihood NSP

The target model uses a mixed observation likelihood containing:

- Binomial or Beta-Binomial channels for test outcomes;
- Beta-family channels for bounded ratios;
- Poisson or Negative-Binomial channels for counts;
- Bernoulli channels for binary events;
- Gamma or Log-Normal channels for latency and heavy-tailed usage measures.

Inference can then be performed using particle filtering or another
non-Gaussian state-space inference method.

 9. Model-Selection Protocol

The final distribution for each channel will not be selected solely from
its semantic type.

Selection will use:

1. empirical support and boundary values;
2. mean-variance relationship;
3. proportion of zeros;
4. overdispersion tests;
5. held-out log-likelihood;
6. AIC/BIC where applicable;
7. posterior predictive checks;
8. residual dependence between channels.

 10. Paper Positioning

The paper will state:

"The Linear Gaussian NSP is used as a transparent and computationally
tractable baseline. Because LLM repair trajectories combine proportions,
counts, binary events, and positive heavy-tailed measurements, the final NSP
uses a mixed observation model with channel-specific likelihoods. Particle
filtering enables sequential inference under these non-Gaussian likelihoods,
while the observation probabilities themselves remain explicitly specified
for particle weighting."