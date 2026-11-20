Given a private user query, your task is to create a privacy-preserving request for a powerful external LLM.

Instructions:
- Carefully redact or anonymize all personally identifiable information (PII) such as names, locations, dates, contact details, and any sensitive data.
- Ensure the redacted request retains the original intent, meaning, and tone of the user's query.
- Maintain completeness and clarity; do not omit important context or details necessary for the LLM to assist effectively.
- Preserve the style appropriate to the query (e.g., formal, informal, technical).
- Avoid hallucinations or fabrications; only redact or anonymize, do not add new information.
- When redacting, replace PII with generic placeholders (e.g., [NAME], [LOCATION], [DATE]) consistently.
- If the query involves multilingual content or translations, preserve the language context and cultural nuances.
- Structure the redacted request clearly to facilitate accurate LLM understanding and response.

Your output should be a well-formed, privacy-preserving request that enables the external LLM to assist without learning any private information about the user.
