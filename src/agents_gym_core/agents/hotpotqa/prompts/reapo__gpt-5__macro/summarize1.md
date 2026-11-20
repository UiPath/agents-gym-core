Task: From the retrieved passages, produce a compact, retrieval-oriented summary for hop 1 that maximizes evidence for answering the multi-hop question.

Instructions:
- Read the question and the passages. Extract only facts that connect named entities and properties relevant to answering the question.
- Output 4–8 bullet facts. Use the format: Entity — relation — value (with years/numbers where present).
- Prioritize: proper names, aliases, titles, occupations, organizations, locations, dates, counts, and nationality.
- Identify potential bridge entities explicitly (e.g., Bridge: <name>), especially people, places, teams, works, or institutions that link the hops.
- Normalize: use canonical names; include a single alias mapping if relevant (Alias: X = Y).
- Exclude fluff, opinions, or unrelated facts.
- Do not answer the question here. Do not speculate.

Produce the field `summary` only.
