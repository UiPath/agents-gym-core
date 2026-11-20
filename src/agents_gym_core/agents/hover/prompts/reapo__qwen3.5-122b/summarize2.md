Given the fields `claim`, `context`, `passages`, produce the fields `summary`.

Your task is to synthesize information from the context and new passages to build a more complete picture of the claim's validity.

Instructions:
1. Review the context from previous hops to understand what has already been established
2. Analyze the new passages for relevant information about the entities and relationships in the claim
3. Update your understanding of which parts of the claim are supported by evidence
4. Identify any remaining gaps in information that need to be filled
5. Note any contradictions or confirmations between the context and new passages
6. Track which of the 3 required documents have been retrieved and which are still missing

The summary should:
- Build upon the previous context
- Integrate new evidence from the passages
- Clearly state what is now confirmed and what still needs verification
- Be precise about entity names and their connections
- List which of the 3 required documents have been found and which are still needed

Focus on extracting the specific document titles that would serve as evidence for the claim. Remember: exactly 3 documents are needed.

IMPORTANT: When analyzing passages, focus ONLY on documents that are directly relevant to the 3 required entities. Ignore irrelevant documents that may appear in the search results.
