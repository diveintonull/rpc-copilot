---
name: gap-analysis
description: Compare an enterprise control description with retrieved regulation evidence and produce a human-reviewed gap matrix. Use for gap_analysis requests that include control_text or equivalent verified control evidence. Do not use for regulation-only questions, clause comparison, missing enterprise context, or automatic compliance determinations.
---

# Control Gap Analysis

## Required inputs

- Require `control_text` supplied by the user or an explicitly connected enterprise evidence source.
- Use only regulation evidence returned by the retrieval Tool.
- Treat missing control information as unknown, not as proof that a control is absent.

## Workflow

1. Extract only controls explicitly supported by `control_text`.
2. Retrieve the regulation requirements relevant to the user's requested scope.
3. Map each requirement to the corresponding current control description.
4. Classify the mapped result as aligned, partial, gap, or unknown.
5. Assign a grounded risk level without turning it into a legal conclusion.
6. Recommend a concrete next action for each partial, gap, or unknown result.
7. Bind each matrix row to the regulation evidence that supports the requirement.
8. Present the entire matrix as a preliminary analysis requiring human confirmation.

## Classification rules

- `aligned`: the supplied control description explicitly satisfies the retrieved requirement. Do not equate aligned with enterprise-wide compliance.
- `partial`: the supplied control covers part, but not all, of the retrieved requirement.
- `gap`: the supplied control explicitly omits or contradicts a required element.
- `unknown`: the supplied information is insufficient to determine alignment. Do not convert silence into a gap.

Begin the `gap` field with the applicable classification label so the rendered matrix preserves the classification.

## Evidence rules

- Ground every requirement in retrieved regulation evidence.
- Keep the enterprise current state separate from the regulation requirement.
- Do not invent an implemented control, missing control, policy, system configuration, audit result, or business context.
- Do not retain a generated evidence reference that was not present in the Tool result.
- Use unknown when evidence about the current state is incomplete.

## Risk and recommendation rules

- Use high only for an explicit gap with material security or regulatory impact supported by the available context.
- Use medium for a material partial implementation or a narrower explicit gap.
- Use low for an aligned item with limited identified residual risk.
- Use unknown when current-state or impact information is insufficient.
- Make recommendations specific and actionable, but do not guarantee that an action will produce compliance.
- Recommend additional evidence collection when the classification is unknown.

## Output

Return one row per mapped requirement with these fields:

```text
requirement
current_state
gap
risk
recommendation
evidence
```

Include the exact regulation source, version, and section number in `evidence`. End the output with an explicit human confirmation requirement.

## Boundary and refusal

Refuse to produce a completed gap matrix when:

- `control_text` or equivalent verified current-state evidence is missing;
- no relevant regulation evidence was retrieved;
- a requirement cannot be bound to regulation evidence;
- the request asks for an automatic legal or compliance determination.

Do not declare that the enterprise is compliant, non-compliant, legal, or illegal. Do not hide unknown information behind a confident recommendation. Require human confirmation before treating any result as final.
