Given the fields `claim`, `summary_1`, produce the fields `query`.

Your task is to create a targeted search query to retrieve the next piece of evidence needed to verify the claim.

Instructions:
1. Analyze the claim to identify what information is still missing after hop 1
2. Review summary_1 to understand what has already been established
3. Identify the specific entity or relationship that needs to be retrieved next
4. Create a query that will retrieve documents about that specific entity/relationship

Query creation guidelines:
- Be specific and targeted - avoid vague queries
- Include key entity names from the claim
- Focus on retrieving documents that directly address the missing information
- The query should help retrieve exactly the documents needed to support the claim
- Think about what document titles would contain the needed information
- Each query should target ONE specific entity/document that is still missing

CRITICAL: The goal is to retrieve exactly 3 supporting documents total. If some have already been retrieved, focus on retrieving only the remaining ones. Create queries that will return the specific document titles needed.

IMPORTANT: When creating queries:
1. Use the exact entity name or a very specific description
2. Avoid generic terms that might return many irrelevant results
3. Focus on retrieving the specific document title needed, not just general information

Remember: You need to retrieve documents about specific entities, not just general information. The query should be designed to return the exact document title needed.
