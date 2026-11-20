You are creating a search query for the second hop of a multi-hop QA task. You have:
- The original question
- Summary from the first hop (summary_1)

**Instructions:**
1. Analyze what information is still needed to answer the question after the first hop
2. Identify the key entity or concept that should be searched for in the second hop
3. Formulate a concise query that will retrieve relevant information

**Query formulation guidelines:**
- If the first hop identified a person, search for their biographical details (birth year, profession, etc.)
- If the first hop identified a work (book, film, album), search for its details (year, creator, etc.)
- If the first hop identified a relationship, search to confirm or find the specific connection
- Make the query specific enough to retrieve the exact information needed
- The query should be a question or phrase that directly targets the missing information

**Examples of good queries:**
- "Peter Gabriel birth year" (when you need a person's birth year)
- "What year was [album] released" (when you need a release year)
- "Who directed [film]" (when you need a director)
- "What is the profession of [person]" (when you need a profession)

Given the fields `question`, `summary_1`, produce the fields `query`.
