Task: Generate a precise BM25 search query for hop-2.

Inputs:
- claim
- summary_1: retrieval-oriented summary from hop-1

Output:
- query: a concise keyword string optimized for retrieval.

Query construction rules:
- Include the main subject and bridge entity by their canonical names; add known aliases in parentheses or as extra tokens.
- Add relation keywords that connect them (e.g., "member of", "located in", "acquired", "capital of", "born in").
- Include critical disambiguators (year, title/role, organization, location) if present.
- Use quotes around multi-word named entities; avoid stopwords.
- Prefer distinctive nouns/proper names over generic terms; avoid long sentences.

Output exactly the query text, nothing else.
