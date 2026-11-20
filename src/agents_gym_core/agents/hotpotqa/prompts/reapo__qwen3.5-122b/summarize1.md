You are summarizing information for a multi-hop QA task. Given a question and a set of Wikipedia passages, extract and summarize the key information that helps answer the question.

**Instructions:**
1. Identify the main entities, people, places, or concepts mentioned in the question
2. Extract relevant facts from the passages that relate to these entities
3. If the passages contain the answer directly, state it clearly
4. If the passages don't contain complete information, note what is missing and what additional information would be needed
5. Be concise but include all relevant details (names, dates, relationships, attributes)

**Output format:**
- Provide a clear summary that captures the essential information from the passages
- If the answer is found, state it explicitly
- If information is missing, identify what specific information is needed to answer the question

Given the fields `question`, `passages`, produce the fields `summary`.
