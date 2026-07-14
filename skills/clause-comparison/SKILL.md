---
name: clause-comparison
description: Compare two regulation clauses or versions with separately retrieved evidence and explicit dimensions. Use for clause_comparison requests that identify or require a left and right clause. Do not use for single-regulation questions, enterprise control gap analysis, legal advice, or comparisons with only one available side.
---

# Clause Comparison

## Required inputs

- Require a left clause and a right clause.
- Require a source, version, and section number for precise lookup whenever available.
- Use comparison dimensions derived from the user's request or an explicit comparison plan.

## Workflow

1. Identify the requested left clause, right clause, versions, and comparison dimensions.
2. Resolve both sides independently with the clause comparison Tool.
3. Confirm that both sides contain retrievable evidence before drafting a comparison.
4. Keep left and right evidence separate throughout the analysis.
5. Compare only the same dimension across both sides.
6. Describe similarities, differences, scope, and obligation strength only when both texts support the statement.
7. Attach evidence identifiers from both sides to every comparison row.

## Evidence rules

- Use evidence from both sides for every claimed similarity or difference.
- Do not use left evidence to prove right-side content or right evidence to prove left-side content.
- Preserve each clause's source, version, and section number.
- Do not invent missing text, dimensions, applicability, or legal effect.
- Distinguish direct textual differences from interpretive observations.

## Version handling

- Label the version on each side even when both clauses come from the same source.
- Compare different versions when the request explicitly asks for a version comparison.
- Do not silently replace an unavailable version with another version.
- Request clarification when a side is version-ambiguous and the ambiguity could change the result.

## Output

Return a table with this structure:

| Dimension | Left | Right | Difference | Evidence |
|---|---|---|---|---|
| Requested dimension | Supported left-side requirement | Supported right-side requirement | Supported similarity or difference | Left and right evidence identifiers |

Follow the table with a short limitation note when scope, version, or applicability remains uncertain.

## Boundary and refusal

Refuse to produce a complete comparison when:

- either the left or right clause cannot be retrieved;
- either side lacks a reliable source, version, section number, or text;
- the requested comparison dimension cannot be supported by both sides;
- the request requires invented text or authority.

Do not rank one clause as stricter unless both sides support that interpretation. Do not declare that an enterprise is compliant, non-compliant, legal, or illegal.
