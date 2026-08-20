NSP Paper Figures Design
This document outlines the planned figures for the manuscript. Figures are split between Paper 1 (Core Methodology & Theory) and Thread 3 (Drift Detection Application).

Part A — Paper 1 / Thread 1 Figures
Figure 1: The NSP Pipeline Architecture
Type: Conceptual Diagram.
Content:
Left: Raw LLM Repair Trace (Prompt, Response, Patch, Test).
Middle: Feature Extraction (21-dim y 
t
​
  vector).
Right: State-Space Model (Latent z 
t
​
  evolving over time t=1→T).
Purpose: Give reviewers an immediate visual understanding of the NSP framework.
Figure 2: Synthetic Parameter Recovery
Type: Line plot with confidence intervals.
Content:
X-axis: Epochs (or Iterations).
Y-axis: Frobenius norm error ∣∣A 
learned
​
 −A 
true
​
 ∣∣ 
F
​
 .
Lines: Gradient MLE, Classical EM, Particle-filter MLE.
Purpose: Prove that the estimators mathematically recover the true hidden dynamics (Answers RQ1).
Figure 3: Finite-Sample Estimation Limits (CRB Gap)
Type: Line plot (Log-Log scale).
Content:
X-axis: Trajectory length T (e.g., 2, 5, 10, 20, 50).
Y-axis: Mean Squared Error (MSE) of hidden state recovery.
Lines: Classical EM vs. Particle Filter.
Reference line: Cramér–Rao Bound (CRB).
Purpose: Demonstrate the theoretical limits and convergence properties of the estimators on short APR trajectories (Answers RQ4).
Figure 4: Model Fit Comparison on CCBench
Type: Bar chart or Box plot.
Content:
X-axis: Models (i.i.d., Discrete HMM, Linear Gaussian NSP, Mixed-Likelihood NSP).
Y-axis: Held-out Average Log-Likelihood.
Purpose: Show that modeling trajectories as a State-Space process significantly outperforms non-sequential or simpler baselines on real data (Answers RQ2).
Figure 5: Cross-Model Consistency Matrix
Type: Heatmap.
Content:
Heatmap showing the similarity of learned transition matrices (A) across different LLMs (e.g., GPT-4o vs. Claude 3.5).
Purpose: Show that NSP captures consistent latent dynamics across different LLM architectures (Answers RQ3).
Part B — Thread 3 / Drift Detection Figures (For future extension paper)
Figure 6: Hidden State Evolution & Drift Onset
Type: Time-series plot.
Content:
X-axis: Repair Round (t).
Y-axis: Latent State Values (p 
t
​
  and s 
t
​
 ).
Highlight: A vertical dashed line at t 
onset
​
  where p 
t
​
  increases but s 
t
​
  drops.
Purpose: Visually demonstrate the "Deceptive Drift" phenomenon (Test-passing-but-wrong).
Figure 7: Drift Detection Trade-offs
Type: ROC Curve or Precision-Recall Curve.
Content:
Curves comparing NSP P 
drift
​
  detector against Heuristic baselines (Pass-rate drop, Token growth).
Purpose: Prove that NSP detects semantic drift earlier and more accurately than naive metrics (Answers RQ-D1, RQ-D2).