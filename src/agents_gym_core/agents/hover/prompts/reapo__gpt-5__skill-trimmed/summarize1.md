Task: From `claim` and `passages`, produce a concise summary that captures ONLY the facts present in `passages` that are relevant to verifying the claim and guiding the next retrieval hop.

Requirements:
- Use ONLY information present in `passages`. Do not add external knowledge or guesses.
- 1–2 sentences max. Be specific and factual.
- Focus on named entities, dates, titles, roles, and direct relations mentioned.
- If a needed piece of information is missing (e.g., a birth date or identity not in passages), explicitly state what is missing to guide the next hop.
- No lists, no bullet points, no meta-comments.

Output: A single 1–2 sentence summary, strictly grounded in `passages`, highlighting what is known and what is missing.
