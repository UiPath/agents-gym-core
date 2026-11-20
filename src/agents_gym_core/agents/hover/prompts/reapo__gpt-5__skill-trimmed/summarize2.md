Task: Using `claim`, `context` (the previous summary), and new `passages`, produce a refined, factual summary to drive the next retrieval.

Requirements:
- Ground strictly in the provided `passages`; do not use outside knowledge.
- Incorporate and update the `context` only where supported by current passages.
- 1–2 sentences max, focused on entities/relations/dates/titles directly relevant to verifying the claim.
- If a key item needed to verify the claim is still missing, state exactly what is missing.
- No lists, no meta output.

Output: One 1–2 sentence factual summary that integrates the new evidence and clearly states any remaining missing piece.
