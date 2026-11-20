Given the fields `question`, `context`, `passages`, produce the fields `summary`.

Your task is to extract and summarize the key facts from the passages and context that are relevant to answering the question. Focus on:
- Identifying the specific entities, names, dates, numbers, and relationships mentioned in the passages.
- Preserving exact names (full names including middle names, nicknames, and formal titles as written in the passages), exact numbers, exact dates, and exact phrases.
- When a passage gives a person's full formal name, always preserve the COMPLETE full name in your summary.
- Combining information from the context (first hop) with new information from the passages (second hop) to build toward the answer.
- If the passages mention multiple entities, clearly distinguish between them.
- Keep the summary factual and concise — include only information that helps answer the question.
- Pay careful attention to what the question is actually asking for and make sure the relevant answer entity is clearly identified.
