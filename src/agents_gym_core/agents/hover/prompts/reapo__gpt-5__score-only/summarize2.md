Task: Refine the summary using the current context and additional passages to expose the bridge needed for the next hop.

Inputs:
- claim
- context: the prior-hop context (e.g., subject and known facts)
- passages: newly retrieved text

Output:
- summary: a retrieval-oriented distillation tailored to generate hop-3 queries.

Guidelines:
- Connect entities from context to new entities in passages; surface the missing link.
- State the relation(s) we need to verify (e.g., affiliation, location, date, role, award, parent company).
- Include canonical names, aliases, and precise disambiguators.
- Keep it short and factual; avoid speculation.

Format:
Brief paragraph or 3–6 bullets mentioning:
- Context entity ↔ New entity
- Relation phrase(s)
- Disambiguators (dates, titles, locations, organizations)
