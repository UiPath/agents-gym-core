Given the fields `claim`, `summary_1`, `summary_2`, produce the fields `query`.

Your task is to create the final search query to retrieve any remaining evidence needed to complete the verification of the claim.

Instructions:
1. Review the claim to identify what evidence is still missing
2. Analyze summary_1 and summary_2 to understand what has been established so far
3. Determine which specific entity or relationship needs to be retrieved in this final hop
4. Create a precise query that will retrieve the remaining supporting documents

Query creation guidelines:
- Be highly specific - this is the final hop, so target exactly what's needed
- Include the key entity names that haven't been retrieved yet
- Focus on retrieving the exact document titles that serve as evidence
- The query should help complete the set of required supporting documents

CRITICAL: Remember that exactly 3 supporting documents are needed total. Track which documents have been retrieved and create a query to get only the remaining ones.

IMPORTANT: Before creating the query:
1. Check which of the 3 required documents have already been retrieved
2. If all 3 are retrieved, indicate that no further query is needed
3. If some are missing, create a query targeting the specific missing entity/document

The query should be designed to return the exact document title needed, not just general information about the topic. Use specific entity names to avoid retrieving irrelevant documents.
