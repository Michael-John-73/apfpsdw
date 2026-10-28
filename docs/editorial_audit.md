# Narrative and claim-evidence audit

We separate three sentence types rather than replace every passive verb or
every figure/table subject with "we".

| Category | Treatment | Example |
|---|---|---|
| External instructions or reviewer-style advice | We replace directives with our stated interpretation or measured limit | "must be read" becomes "We use this proposition to interpret..." |
| Our experimental actions hidden in passive voice | We identify the actor explicitly | "forecasts are saved" becomes "We save input-only forecasts before obtaining labels" |
| Mathematical definitions, algorithm operations and display descriptions | We retain conventional technical subjects | "Table reports...", "K counts errors", and algorithm input/output steps |

We checked manuscript prose and public source/document text for the phrases
"must not be added", "Select the pair", "design approval" and
"eligibility certificate". We found none in the active manuscript or
the released pipeline modules. We distinguish that absence from our broader
review of actual directive and passive constructions.

We revised output interpretations in the public `operations.py` so our
limitations state what we measured and inferred rather than tell a reader
what to report. We retain software validation messages and algorithm
imperatives where they specify execution behavior rather than research prose.

## Thesis, hypotheses and experiments

| Claim | Original criterion or contrast | Evidence and retained limit |
|---|---|---|
| Detection does not establish generated origin | H1 public insertion; S-F paired RF-R0 | H1 near-unity prediction fails; positive public insertion still provides a counterexample to origin authentication |
| Reliability depends on implementation and channel | Source tracing, H2, paired dtype/detail controls, S-T | H2 robustness prediction fails; all native S-T numerical margins pass but the original duplicate audit hold remains |
| A secret string is not image-bound authentication | H3 restricted blind attacker; later reference access | Blind control supports its limited claim; one-reference successes prevent a general security inference |
| Prediction needs calibration and cost evidence | H_BP and H_BT frozen criteria | H_BP remains inconclusive; H_BT meets locked criteria with severe transfer undercoverage and 13.10% greater prediction time |

We retain the chronological distinction between original hypotheses and later
frozen extensions. We do not present diagnostics as new confirmation, numerical
checks as scientific support, or the safety-replacement explanation as a
replacement S-T verdict.
