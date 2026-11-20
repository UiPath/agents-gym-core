Task: Generate a targeted second-hop query to retrieve the specific fact needed.

Inputs: question, summary_1
Output: query

Instructions:
- Identify the bridge entity from summary_1; include a disambiguator if needed.
- Include the exact relation/attribute needed for hop 2. Consider synonyms (e.g., "spouse"/"husband", "birth place"/"born in").
- Prefer quoted phrases and 1–2 additional keywords from the question.
- Keep it concise and specific; avoid stopwords.

Format:
"<Bridge Entity> (<disambiguator>)" "<relation or attribute>" <keyword1> <keyword2>
