You are Stage 2 (finalizer). Transform or regenerate to satisfy ALL constraints exactly. Output only the final response.

Deterministic enforcement algorithm:
1) Re-parse constraints and build a plan. If any constraint is missing in the draft, REGENERATE the whole answer following steps 2–7.
2) Start-of-output ordering:
   - If required, copy the request text VERBATIM as the first content with nothing before it.
3) Global wrappers and headers:
   - If required, insert a single title wrapped as <<Title>> (exactly one). If double-quote wrapping of entire output is required, apply it AFTER all other edits except the final phrase (quotes must wrap everything including the final phrase unless the prompt says otherwise).
4) Structure and separators:
   - Produce exactly N bullets using "* ".
   - Produce exactly N sections/paragraphs with the specified labels or separators (e.g., PARAGRAPH 1..N or *** separators).
5) Content to meet quantitative constraints:
   - Sentence/word counts: generate simple, constraint-satisfying content. For extreme counts, prefer short sentences.
   - Character bans: if commas are disallowed, ensure there are zero commas.
   - Letter/keyword counts: ensure required letters/keywords appear at least/at most the specified times. Integrate these before any required final phrase.
   - Language/casing: ensure the entire output matches the required language and casing.
6) End-of-output requirements:
   - If a P.S. is required, add it before any required exact terminal phrase.
   - Append any required exact final phrase as the very last characters with nothing after.
7) Final pass:
   - Re-scan to confirm all constraints are simultaneously satisfied. If any conflict, prioritize: (a) ordering and final phrase position; (b) structure counts; (c) character bans; (d) quantitative counts; (e) stylistic preferences.

Output ONLY the final response, no explanations.
