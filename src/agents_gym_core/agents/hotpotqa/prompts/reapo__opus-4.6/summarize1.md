Given the fields `question`, `passages`, produce the fields `summary`.

Your task is to extract and summarize the key facts from the passages that are relevant to answering the question. Focus on:
- Identifying the specific entities, names, dates, numbers, and relationships mentioned in the passages.
- Preserving exact names (full names including middle names, nicknames, and formal titles as written in the passages), exact numbers, exact dates, and exact phrases.
- When a passage gives a person's full formal name (e.g., "Varazdat Samuel 'Varaz' Samuelian" or "Mary Barbara Hamilton Cartland"), always preserve the COMPLETE full name in your summary.
- Noting which facts are directly relevant to the question and which entity/passage they come from.
- If the passages mention multiple entities, clearly distinguish between them.
- Keep the summary factual and concise — include only information that helps answer the question.
- Pay careful attention to what the question is actually asking for (a person, a place, a film title, an occupation, etc.) and make sure the relevant answer entity is clearly identified in your summary.
