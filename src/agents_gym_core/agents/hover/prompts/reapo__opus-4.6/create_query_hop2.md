You are generating a BM25 keyword search query to retrieve a specific Wikipedia page.

Given the fields `claim` and `summary_1`, produce the field `query`.

Rules:
- Look at the "NEED TO RETRIEVE" list in summary_1 and pick the FIRST (most important) entity
- Output ONLY 2-5 search keywords matching that entity's likely Wikipedia article title
- Use exact proper nouns as they would appear in a Wikipedia title
- Prioritize retrieving PEOPLE and specific NAMED WORKS over generic categories
- Do NOT ask questions — just output keywords
- Do NOT search for entities whose pages are already retrieved
