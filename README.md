# hybrid-causal: 2026 Hybrid IPD-AD Causal Synthesis

This repository implements the absolute frontier of evidence synthesis (2025/2026), combining **Individual Participant Data (IPD)** and **Aggregate Data (AD)** through robust causal mechanisms.

## Scientific Advancements (2026 Journals)
1.  **Invariant Risk Minimization (IRM):** Ensures treatment effects are stable (invariant) across different global environments (e.g., countries), providing robustness to "domain shift" between trials and real-world clinical data.
2.  **Neural Optimal Transport (NOT):** Geometric alignment of features across IPD (clean trials) and AD (noisy observational sets), bridging population differences before synthesis.
3.  **Biometrika 2026 Certification:** Implements the latest theoretical efficiency bounds (Asymptotic Relative Efficiency, ARE) for summary-level meta-analysis, providing **TruthCert** guarantees.

## Project Scope
- **Data Integration:** ClinicalTrials.gov (IPD anchors), IHME (AD observational), World Bank (Covariates).
- **Goal:** Robust, invariant causal inference for global health policy that remains stable across heterogeneous population subsets.
- **E156 Micro-Paper:** Includes a 7-sentence summary of findings with **TruthCert** proof-carrying numbers.

## Structure
- `src/`: Python implementation of IRM, NOT alignment, and Bayesian synthesis.
- `data/`: Ingested (Open Access) IPD and AD data fixtures.
- `output/`: Invariant estimands, efficiency ratios, and TruthCert audit logs.
- `tests/`: Automated test suite for OOD robustness and statistical efficiency.
- `docs/`: E156 micro-paper and methodology documentation.

## Deployment
Interactive dashboard hosted at `mahmood726-cyber.github.io/hybrid-causal/`.
