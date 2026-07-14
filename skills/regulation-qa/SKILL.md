---
name: regulation-qa
description: Answer questions about requirements in one regulation or regulation set using retrieved evidence and numbered citations. Use for regulation_qa requests about what a regulation requires. Do not use for comparing two clauses, analysing enterprise control gaps, legal advice, or unsupported requests.
---

# Regulation QA

## Required inputs

- Use the user's regulation question.
- Use only regulation evidence returned by the retrieval Tool.
- Respect a source or version explicitly requested by the user.

## Workflow

1. Identify the requested subject, source, jurisdiction, and version.
2. Retrieve the relevant regulation clauses with the Tool.
3. Check every clause's source, version, section number, and text.
4. Handle any version conflict before drafting the answer.
5. Answer only the requirements supported by retrieved evidence.
6. Cite every factual claim with the corresponding numbered citation such as `[1]`.
7. State any evidence or applicability limitation separately from the factual answer.

## Evidence rules

- Use only retrieved evidence; do not rely on unstated regulatory knowledge.
- Keep each citation attached to the claim it supports.
- Do not invent a source ID, version, section number, effective date, or quotation.
- Do not use one clause to support a materially broader requirement.
- Treat an inference as an inference and identify the evidence behind it.

## Version conflict handling

- Use the exact version when the user specifies one.
- Do not merge requirements from different versions into one statement.
- When multiple versions conflict and the user did not choose one, disclose the version conflict and request clarification or present the versions separately.
- Do not decide which version legally applies without sufficient applicability information.

## Output

Return:

1. A direct answer.
2. Numbered citations for every factual requirement.
3. A version note when version choice or conflict matters.
4. A limitation note when evidence or applicability is incomplete.

## Boundary and refusal

Refuse to provide a grounded regulatory answer when:

- no relevant regulation evidence was retrieved;
- the requested source or version is unavailable;
- the available evidence does not support the requested conclusion;
- citations cannot be bound to the factual claims;
- the request asks the Agent to ignore evidence or invent authority.

Do not declare that an enterprise is compliant, non-compliant, legal, or illegal. Do not present the answer as legal advice.
