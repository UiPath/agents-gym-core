You generate a BM25 keyword query for hop 3 of a multi-hop Wikipedia retrieval pipeline (HoVer).

Inputs:
- claim
- summary_1 and summary_2 (from prior hops)

Goal:
Output a single, highly discriminative keyword query string that targets the final evidence page(s) needed to verify the claim.

Query construction rules:
- Include: (a) the claim’s target entity, (b) the decisive bridge entity from summary_2 (or summary_1 if stronger), and (c) 1 relation keyword that ties them.
- Prefer canonical page types and titles: film, album, novel, award, constituency, city, etc.
- Add disambiguators (year, role, title, location) where present; include a year if available.
- Exclude list-like or generic terms: list, discography, filmography, franchise, category, series.
- No punctuation or boolean operators. Lowercase words, 6–9 tokens total.

Output field:
- query: one line with the keywords only (no extra text).

Format examples (synthetic):
- Example A: film composer link → desert wind film alex chen composer 2003
- Example B: politician constituency link → maria lopez politician green valley constituency
- Example C: band label link → the night trains band starlight records label
