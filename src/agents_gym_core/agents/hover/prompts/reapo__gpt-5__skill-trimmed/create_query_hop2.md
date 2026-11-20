Task: From `claim` and `summary_1`, produce a single focused search query string to retrieve the one missing fact identified.

Hard constraints:
- Include at least one exact quoted multi-word entity from the claim or summary (e.g., "Handsome Harry", "Child 44"). Prefer including BOTH entities implicated by the claim when relevant, joined with AND.
- Use AND between key terms. No commas. No periods.
- Add a property keyword only if it sharpens intent (e.g., born, birth, director, producer, soundtrack, film).
- Avoid category words and broad scopes: do not include list, franchise, series, discography, awards, episodes.
- Max ~8 tokens/phrases. Output must be only the query string.

Output: A single high-precision query string anchored on the exact entities and the needed property.
