Given the fields `question`, `summary_1`, `summary_2`, produce the field `answer`.

Strict output rules (critical for Exact Match evaluation):
- Output ONLY the final answer string. No reasoning, no prefixes/suffixes, no punctuation, no quotes.
- Provide the minimal, canonical span that answers the question.
- Match number and plurality as asked (e.g., "dog" vs "dogs").
- Yes/No questions: output exactly `yes` or `no` (lowercase).
- Years: if the question asks "what year", output only the 4-digit year. Dates: use the canonical form in the sources (e.g., "August 2, 1973").
- Numbers: preserve comma separators if that is the canonical figure (e.g., "267,785").
- Names/titles/places: use the full canonical form from the sources (e.g., "Lipscomb University", "Robert Lindstedt").
- Nationalities/demonyms: match the canonical term (e.g., "English" vs "British" as appropriate to the source or question).
- When multiple plausible answers exist, choose the one that directly and exactly answers the question with the most specific, canonical phrasing present in the summaries.

Process:
1) Use `summary_1` and `summary_2` to determine the precise answer.
2) Output only the answer text that satisfies the rules above.
