Task: From `claim`, `summary_1`, and `summary_2`, produce a single focused search query string targeting the final missing fact.

Hard constraints:
- Include an exact quoted multi-word entity; when the claim links two entities, include both with AND if possible.
- Join key terms with AND; include one property keyword if useful (born, birth, director, producer, album, film).
- Avoid broad/category terms: list, franchise, series, discography, awards, episodes.
- Max ~8 tokens/phrases. Output must be only the query string.

Output: One compact, precise query that directly targets the unresolved fact.
