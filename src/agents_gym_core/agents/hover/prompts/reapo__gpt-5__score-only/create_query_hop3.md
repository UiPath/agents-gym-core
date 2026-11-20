Task: Generate a precise BM25 search query for hop-3 using all known context.

Inputs:
- claim
- summary_1
- summary_2

Output:
- query: a concise keyword string optimized for retrieval.

Query construction rules:
- Combine the subject(s) from summary_1 with the new entity/entities in summary_2.
- Include relation phrases (e.g., affiliation, ownership, birthplace, headquarters) that match the claim.
- Add strong disambiguators (dates, titles, locations, organizations, middle initials) when available.
- Quote multi-word entity names; include key aliases if likely.
- Keep under ~12 tokens; prioritize distinctive tokens over filler.

Output exactly the query text, nothing else.
