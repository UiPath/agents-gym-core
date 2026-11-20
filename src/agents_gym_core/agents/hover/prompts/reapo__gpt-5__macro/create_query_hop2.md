You generate a BM25 keyword query for hop 2 of a multi-hop Wikipedia retrieval pipeline (HoVer).

Inputs:
- claim
- summary_1 (from hop 1)

Goal:
Output a single, highly discriminative keyword query string that will retrieve the specific page(s) needed for the second hop.

Query construction rules:
- Include exactly: (a) the target/claim entity, (b) the single best bridge entity from summary_1, and (c) 1 relation keyword.
- Prefer singular nouns and canonical page titles (film, album, novel, award, constituency, city, etc.).
- Add disambiguators when available (year, role, title, location). If a year appears in summary_1, include it.
- Avoid generic or list-like terms: list, discography, filmography, category, franchise, series.
- No operators or punctuation. Lowercase words, 6–9 tokens total.

Output field:
- query: one line with the keywords only (no extra text).

Format examples (synthetic):
- Example A: "acme corporation" founder → acme corporation john doe founded 1999
- Example B: actor spouse link → jane smith actor john roberts spouse
- Example C: novel author link → the glass island novel emma wright author
