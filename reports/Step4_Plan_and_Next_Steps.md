7. Official Next Steps (Step 4: Raw Trace Acquisition & Drift Validation)
The next phase of the project involves acquiring complete LLM execution files, including Prompts, Responses, Patches, and Test results for each Round from the execution server.

Upon acquisition, the workflow will proceed as follows:

Data Ingestion: Raw files will be converted to the NSP-Trace v1.0 standard and validated.
Feature Extraction: Real multi-step sequences (T≫2) will be constructed by extracting the 21-dimensional feature vectors (y t​) for each actual round.
Model Fitting: The implemented Classical EM and Particle-filter MLE algorithms will be fitted onto these real multi-step trajectories to estimate the latent state evolutions.
Semantic Drift Validation: To prove that the estimated hidden states genuinely represent "Semantic Drift", the model's output will be correlated against independent semantic metrics. These validation anchors include:
Semantic Completeness scores.
Human audit annotations.
Ground-truth patch comparisons.
Behavioral contract satisfaction levels.
Currently, acquiring these raw execution traces is the number one priority. Concurrently, the formal definition of Drift and its evaluation protocol will be finalized.