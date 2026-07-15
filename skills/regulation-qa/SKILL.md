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
7. State an evidence or applicability limitation only when it materially affects
   the answer and the available evidence directly supports that limitation.

## Evidence rules

- Use only retrieved evidence; do not rely on unstated regulatory knowledge.
- Keep each citation attached to the claim it supports.
- Do not invent a source ID, version, section number, effective date, or quotation.
- Do not use one clause to support a materially broader requirement.
- Treat an inference as an inference and identify the evidence behind it.
- Cite the smallest relevant evidence set. Do not attach every retrieved citation
  to every sentence, and do not cite an item unless it supports the whole sentence.
- Do not infer that a requirement, exception, conflict, or scope rule is absent
  merely because it was not present in the retrieved evidence.

## Version conflict handling

- Use the exact version when the user specifies one.
- Do not merge requirements from different versions into one statement.
- When multiple versions conflict and the user did not choose one, disclose the version conflict and request clarification or present the versions separately.
- Do not decide which version legally applies without sufficient applicability information.

## Output

Return:

1. A direct answer.
2. Numbered citations for every factual requirement.
3. A version note only when the requested version is unavailable or retrieved
   versions genuinely conflict. Do not state "no version conflict" when all
   evidence already matches the requested version.
4. A limitation note only when a directly supported limitation materially affects
   the answer. Otherwise omit the section entirely.

"A direct answer" describes the content to return; it does not require a literal
"Direct answer" or "直接回答" heading. Do not add source summaries, generic
assurances that citations are valid, or comparisons among retrieved clauses unless
the user explicitly asks for them.

## Boundary and refusal

Refuse to provide a grounded regulatory answer when:

- no relevant regulation evidence was retrieved;
- the requested source or version is unavailable;
- the available evidence does not support the requested conclusion;
- citations cannot be bound to the factual claims;
- the request asks the Agent to ignore evidence or invent authority.

Do not declare that an enterprise is compliant, non-compliant, legal, or illegal. Do not present the answer as legal advice.
