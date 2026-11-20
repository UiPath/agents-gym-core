You are producing the final answer for a multi-hop QA task. You have:
- The original question
- Summary from the first hop (summary_1)
- Summary from the second hop (summary_2)

**Instructions:**
1. Review both summaries to extract the complete answer
2. Combine information from both hops if needed
3. Produce a concise, direct answer that exactly matches what the question asks for

**Answer format guidelines:**
- For "what" questions: Provide the specific entity name or description
- For "who" questions: Provide the person's full name
- For "when" questions: Provide the year or date
- For "yes/no" questions: Answer with just "yes" or "no"
- For comparative questions: Provide the specific entity that satisfies the condition
- Avoid verbose explanations - just give the answer
- If the answer is a description (like "the eponymous mystery writer protagonist"), use that exact phrasing

**Important:**
- The answer should be as concise as possible while still being correct
- Do not add extra context or explanations unless the question specifically asks for them
- Match the expected answer format (e.g., if the answer is a year, just give the year)

Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`.
