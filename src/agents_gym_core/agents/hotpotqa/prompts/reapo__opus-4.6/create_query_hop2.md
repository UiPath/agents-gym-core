Given the fields `question`, `summary_1`, produce the fields `query`.

Your task is to generate a search query for the second hop of a multi-hop question. The summary_1 contains facts from the first retrieval. You need to identify what additional information is still needed to fully answer the question, and create a targeted search query to find that missing information.

Guidelines:
- Identify the bridge entity or missing fact from summary_1 that connects to the final answer.
- Create a specific, focused query that will retrieve the passage needed for the second hop.
- Do NOT simply restate the original question — instead, query for the specific missing piece of information.
- Use entity names and specific terms from summary_1 to make the query precise.
- The query should be a short, keyword-rich search phrase (like a Wikipedia article title or search engine query), NOT a full question or sentence.
- If the summary_1 already contains all the information needed to answer the question, still generate a query that could confirm or provide additional details about the key entity or fact.
