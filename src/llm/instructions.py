DECOMPOSITION_INSTRUCTIONS = """
Goal: Split the question into minimal atomic subqueries. Each subquery targets exactly one company and one quarter (single 10-Q).

Rules:
- Include company and exact quarter/year in every subquery.
- Allowed quarters: 2022 Q3, 2023 Q1, 2023 Q2, 2023 Q3 (never Q4). Quarters must be "Q1"|"Q2"|"Q3"; years 2022|2023.
- Multi-document intent → one subquery per quarter (chronological) across [2022 Q3, 2023 Q1, 2023 Q2, 2023 Q3]. Triggers: "these 10-Qs", "quarterly reports", "over time", "across periods/quarters", plural "reports/filings", requests to summarize/compare legal risks/accounting.
- "latest" → 2023 Q3; "previous" → 2023 Q2. Single-quarter asks → one subquery.
- One metric/topic per subquery.

Normalize metrics for retrieval: total net sales; operating expenses (R&D and SG&A); gross margin percentage; net cash provided by operating activities; shares repurchased and dollar amount; debt (incl. commercial paper) and interest expense; effective tax rate; key legal proceedings and contingencies / key risk factors / significant accounting policy changes; stated implications/expected impact (when the question asks for effects/implications).

Output ONLY JSON:
{"subqueries": ["<subquery 1>", "<subquery 2>", "..."]}
"""


ROUTING_INSTRUCTIONS = """
Infer route for one subquery: year, quarter, company ticker.

Allowed: year ∈ {2022, 2023}; quarter ∈ {"Q1","Q2","Q3"} (never Q4); company ∈ {AAPL, AMZN, INTC, MSFT, NVDA}. Map: Apple→AAPL, Amazon→AMZN, Intel→INTC, Microsoft→MSFT, NVIDIA→NVDA.

Rules: use explicit quarter/year if present; "latest/most recent"→2023 Q3; "previous quarter"→2023 Q2; map any clear date to the allowed quarter; never output Q4.

Output ONLY JSON:
{"year": 2022 or 2023, "quarter": "Q1"|"Q2"|"Q3", "company": "AAPL"|"AMZN"|"INTC"|"MSFT"|"NVDA"}
"""


SINGLE_ANSWER_GENERATION_INSTRUCTIONS = """
Answer using ONLY the provided single-company, single-quarter context. Be thorough before concluding "not found".

Extraction:
- Scan all text; don’t stop early. Accept synonyms: revenue/total net sales; opex/operating expenses; SG&A; R&D; operating cash flow/net cash provided by operating activities; buyback/share repurchase; debt/borrowings/notes/commercial paper; interest expense/cost; gross margin/gross margin %.
- Prefer period-correct figures: quarter (3-month) over YTD unless YTD requested; company totals over segments unless segment asked.
- Preserve units/currency/commas; convert parentheses to negatives when needed.
- If multiple candidates, choose best-aligned; optionally note (e.g., "3-month figure").
- Provide partial answers if only some details exist.

Style:
- Concise: 1–2 sentences; short bullets for multiple values; mention quarter/year if helpful.
- If the question requests downstream effects (impact, implications, drivers, causes, outlook, risks), append one compact, grounded addendum capturing the stated effect/implication/driver from the context (e.g., "no material adverse effect expected", "increase driven by X", "subject to FX risk"). No citations/speculation.

Fallback: Only if nothing relevant appears anywhere, reply: "I couldn't locate this in the provided context."
"""


MULTI_ANSWER_GENERATION_INSTRUCTIONS = """
You are given multiple sub-answers (each from a specific company-quarter) plus the original question. Synthesize a final answer using only that content.

Style:
- Numeric trends/comparisons: compact bullets, one line per quarter (label + value), then a one-sentence conclusion.
- Qualitative multi-quarter (legal, risks, accounting): group bullets by quarter (e.g., "Q3 2022:") listing key points; add a brief note on changes over time.
- If the question requests downstream effects (impact, implications, drivers, causes, outlook, risks), end with a single compact line summarizing the stated effect/implication/driver across quarters (e.g., "no material adverse effect expected", or specific potential outcomes like fines/injunctions/governance changes) grounded in the context.
- Keep exact figures/units; avoid duplication; prefer most specific statements.

Sources: End with "SOURCE(S): " + comma-separated unique filenames from sub-answers (detect "Source: <filename>").

If insufficient information, say so concisely.
"""
