Task: Using the question, first-hop context, and additional passages, extract complementary facts and connect the hops.

Inputs: question, context (from summarize1), passages
Output: summary

Instructions:
- Infer the expected answer type from the question (person, location, organization, date, number).
- Resolve coreference and aliases to canonical names.
- Use the bridge entity to find the second-hop entity or attribute.
- Extract only facts that help answer the question; include provenance [p#].
- Prefer short spans (names, titles, dates, numbers). Normalize dates as needed.
- If facts conflict, list both and mark as "conflict" with [p#] for each.
- State the candidate answer explicitly when derivable, matching the inferred answer type.

Format summary as:
- Bullet facts ≤ 20 words: "[p#] <entity> — <relation> — <value>"
- End with: "Candidate answers: <span1>; <span2>" or "Answer candidate: <span>".
