Given the fields `claim`, `passages`, produce the fields `summary`.

Your task is to analyze the claim and identify the key entities and relationships that need to be verified. The claim typically involves a multi-hop reasoning task where you need to connect multiple pieces of information.

Instructions:
1. Identify ALL named entities in the claim (people, places, organizations, works, etc.)
2. Determine what relationships between entities need to be verified
3. Note which entities are already confirmed in the provided passages
4. Identify what information is missing and needs to be retrieved in subsequent hops
5. List the specific document titles that would serve as evidence for each entity

CRITICAL: You must identify exactly 3 key entities/documents that are needed to verify the claim. These are the documents that should be retrieved.

The summary should be concise but capture:
- The main entities involved (list all 3 required entities)
- The relationship being claimed
- What evidence from the passages supports the claim
- What specific document titles need to be retrieved for each entity

Be very specific about entity names and their roles in the claim. The goal is to retrieve exactly 3 supporting documents.

IMPORTANT: Focus on identifying the EXACT document titles needed. For multi-hop claims, the 3 documents typically include:
1. A document about the first entity mentioned in the claim
2. A document about the second entity or the relationship
3. A document that connects the entities or provides the final piece of evidence

Avoid including irrelevant documents in your analysis.
