Task: Integrate hop-2 passages with the question and prior context to produce a reasoning-ready summary for answer extraction.

Instructions:
- Inputs: question, prior context (summary_1), new passages (passages).
- Resolve coreference and align aliases between summary_1 and the new passages.
- Extract only facts that complete the chain needed to answer the question.
- Output 4–8 bullet facts, format: Entity — relation — value.
- Emphasize: who/what/where/when relationships, numeric values, titles, and exact names as written in the passages.
- If multiple candidates exist, keep those directly supported by text; note conflicts briefly (Conflict: ...), but do not speculate.
- Keep it concise and factual. No conclusions.

Produce the field `summary` only.
