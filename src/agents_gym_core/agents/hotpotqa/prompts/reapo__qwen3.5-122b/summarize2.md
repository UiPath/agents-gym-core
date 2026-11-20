You are synthesizing information for a multi-hop QA task. You have:
- The original question
- Context from the first hop (summary_1)
- New passages from a second search

**Instructions:**
1. Review the context from the first hop to understand what information was already gathered
2. Analyze the new passages to find information that fills the gaps identified in the first hop
3. Combine information from both sources to form a complete answer
4. If the new passages contain the missing information, extract it clearly
5. If information is still missing, note what is needed

**Key considerations:**
- Look for connections between entities mentioned in the first hop and the new passages
- Pay attention to dates, names, relationships, and specific attributes
- The answer may require combining facts from multiple passages

Given the fields `question`, `context`, `passages`, produce the fields `summary`.
