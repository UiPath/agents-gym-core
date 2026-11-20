Task: Produce the final concise answer.

Inputs: question, summary_1, summary_2
Output: answer

Instructions:
- Infer the answer type from the question and ensure the output matches that type.
- Derive the answer from the intersection of facts across summaries.
- Prefer answer strings that exactly appear in the passages.
- If multiple candidates, choose the one supported by both hops; if tie, select the most specific span.
- Return only the short answer (no explanation). Strip parentheses and trailing punctuation.
- Normalize:
  - Names: full canonical form if available.
  - Numbers: digits.
  - Dates: ISO (YYYY-MM-DD or YYYY-MM; or YYYY if only year).
- Yes/No questions: answer "yes" or "no" strictly based on evidence.
- If uncertain, return "unknown".
