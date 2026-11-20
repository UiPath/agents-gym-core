Task: Output the exact gold-style short answer using summary_1 and summary_2.

Exact-span rules:
- Output ONLY the answer text. No extra words, punctuation, or explanations.
- Choose the exact surface span as it appears in the evidence. Prefer the most specific, canonical span used in the passages.
- People: include full given/middle/surnames and quoted nicknames if present (e.g., Robert Siers "Rob" Cavallo). Do not drop parts.
- Organizations/places/works: include full official names (e.g., Belair railway station, AT&T, The New York Times). Do not add descriptors (e.g., avoid "American" if not part of the name).
- Abbreviations vs expanded forms: pick the form used in the supporting passage(s) that directly answer the question (e.g., PATH not PATH system if the passage uses PATH as the name).
- Dates: include full required specificity seen in the passage (e.g., May 15, 1940; August 2, 1973). Years alone only if that is the natural span.
- Numbers: digits only (e.g., 267,785 if commaed in the passage; 34). Keep commas if present in the evidence.
- Categories: match number and form (e.g., professional wrestler not professional wrestlers) as used in the passage.
- yes/no: exactly "yes" or "no" (lowercase).

Produce the field `answer` only.
