Task: Summarize the key facts from the retrieved passages that are most useful for multi-hop evidence retrieval of the claim.

Inputs:
- claim: a factual statement to verify
- passages: text snippets retrieved in hop-1

Output:
- summary: a compact, retrieval-oriented summary. Focus on bridge entities and relations.

Guidelines:
- Extract and list the main subject(s), bridge entities, and the relation(s) connecting them.
- Include canonical names, aliases, and salient attributes (dates, titles, locations, organizations).
- Prefer facts that help form a precise next-hop query; omit commentary.
- Be specific: include unique identifiers or disambiguators when present.
- If passages disagree, note the dominant consensus and any key alternative alias.

Format:
Return a brief paragraph (2–4 sentences) or 3–6 bullets that explicitly mention:
- Subject → Bridge entity → Relation
- Names/aliases
- Constraints (dates, locations, roles)
