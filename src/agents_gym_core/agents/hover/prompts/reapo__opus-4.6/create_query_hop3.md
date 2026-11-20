You are generating a final BM25 keyword search query to retrieve the last missing Wikipedia page.

Given the fields `claim`, `summary_1`, and `summary_2`, produce the field `query`.

Rules:
- Look at the "NEED TO RETRIEVE" in summary_2 and target that specific entity
- Output ONLY 2-5 search keywords matching that entity's likely Wikipedia article title
- Use exact proper nouns as they would appear in a Wikipedia title
- Prioritize retrieving PEOPLE and specific NAMED WORKS over generic categories
- Do NOT ask questions — just output keywords
- Do NOT search for entities whose pages are already retrieved
