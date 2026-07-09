# Corpus Sources

Raw files live in `data/raw/` (git-ignored). This file is the tracked record of provenance.

| File | Title | Version / Date | Jurisdiction | Source | Downloaded |
|---|---|---|---|---|---|
| GBT+22239-2019.pdf | Information security technology — Baseline for classified protection of cybersecurity (MLPS 2.0) | GB/T 22239-2019 | China | National Public Service Platform for Standards Information (openstd.samr.gov.cn) | 2026-07-09 |
| GBT+35273-2020.pdf | Information security technology — Personal information security specification | GB/T 35273-2020 | China | National Public Service Platform for Standards Information (openstd.samr.gov.cn) | 2026-07-09 |
| CELEX_32016R0679_EN_TXT.pdf | General Data Protection Regulation (GDPR), Regulation (EU) 2016/679 | Adopted 2016-04-27, in force 2018-05-25 | EU | EUR-Lex (eur-lex.europa.eu, CELEX 32016R0679) | 2026-07-09 |
| cybersecurity-law.html | Cybersecurity Law of the PRC | Amended text published 2025-12-29 | China | CAC (cac.gov.cn, c_1768735112911946) | 2026-07-09 |
| data-security-law.html | Data Security Law of the PRC | Adopted 2021-06-10, in force 2021-09-01 | China | CAC (cac.gov.cn, c_1624994566919140) | 2026-07-09 |

Count: **5 documents** — meets the Day 1 target (>= 5).

## Parsing paths

Raw sources are converted to Markdown in `data/parsed/` by three scripts:

- `ingest/parse.py` — text-clean PDFs via pymupdf4llm (GDPR).
- `ingest/parse_mineru.py` — Chinese GB/T standard PDFs via MinerU OCR (GB/T 22239, GB/T 35273); pymupdf yields mojibake on these because their embedded fonts have no ToUnicode map.
- `ingest/parse_html.py` — HTML statutes via BeautifulSoup (`main-content` container) + markdownify (Cybersecurity Law, Data Security Law).

## Notes

- ISO/IEC 27001 full text is paid and therefore excluded; GB/T 22239 (MLPS 2.0) is the primary corpus.
