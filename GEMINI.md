# GEMINI.md — hybrid-causal Research Pipeline Rules

## Purpose
Ship a state-of-the-art **Hybrid IPD-AD Causal Synthesis** using **Invariant Risk Minimization (IRM)** and **Neural Optimal Transport (NOT)** as an **E156 micro-paper + GitHub repo + interactive dashboard**.

## Session Workflow
Before you use any tool or make changes, briefly say what you're about to do, then do it. After each tool call, summarize what you found and what's next.

## Statistical Framework: 2026 Journal Methods
- **Invariant Risk Minimization (IRM):** Identifying treatment effects that are invariant across global environments (environments = countries/regions).
- **Neural Optimal Transport (NOT):** Geometric alignment to bridge Individual Participant Data (IPD) from trial anchors and Aggregate Data (AD) from IHME/WHO.
- **Biometrika 2026 Efficiency:** Certification of summary-level efficiency vs IPD benchmarks.

## Non-negotiables
1. **OA-only**: no paywalls.
2. **No secrets**: redact before logs.
3. **Memory ≠ evidence**: certified claims cite evidence locators + hashes.
4. **Fail-closed**: if validation incomplete, REJECT + reasons.
5. **Determinism**: fixed seeds (seed=42), stable sorting.

## TruthCert (proof-carrying numbers)
- **Every number must be certified.**
- Evidence locator + hash + transformation + validator.
- Asymptotic efficiency ratio (ARE) from Biometrika 2026 MUST be reported.

## Quality Loop
- **Fix ALL issues in one pass.**
- **Test after EACH change.**

## SHIP Ritual
1. Run full test suite.
2. Perform TruthCert audit.
3. Deploy HTML dashboard to GitHub Pages.
4. Update master `INDEX.md` and workbook.

## Platform Defaults
- Python-first (PyMC 5.28.2).
- Offline-first tests + fixtures.
