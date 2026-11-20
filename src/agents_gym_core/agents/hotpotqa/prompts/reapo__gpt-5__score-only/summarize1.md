Task: From the provided passages, extract only the facts needed to answer the multi-hop question.

Inputs: question, passages
Output: summary

Instructions:
- Identify the bridge entity and key relations mentioned in the question.
- Capture aliases and disambiguators (occupation, nationality, year) for common names.
- Extract short, atomic facts relevant to the entities/relations.
- Include passage provenance using [p#].
- Prefer exact names, titles, dates, locations, counts. Normalize dates to ISO (YYYY-MM-DD when precise; YYYY or YYYY-MM otherwise).
- Avoid narrative and long quotes.

Format summary as:
- Fact bullets, each ≤ 20 words: "[p2] Bridge entity: <name> (<disambiguator>) — <relation> — <value>"
- End with one line: "Bridge candidates: <name1>; <name2>" if uncertain; otherwise "Bridge: <name> (<disambiguator>)".
