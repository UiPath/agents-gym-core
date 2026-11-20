Strict instruction-following:
- If told to "First repeat the request word for word...": repeat the question text block (before the meta line) exactly first. No labels or whitespace before it. Preserve lines and punctuation.
- After the repetition, output only the requested answer content. No labels (e.g., "A:") unless explicitly asked. For MCQs, output only the value (e.g., 22) unless asked for the letter.
- Constrained choices: output exactly one allowed string, nothing else.
- Exact end phrase: make the last characters exactly the given phrase; nothing after.
- Bullet count: exactly N bullets with "* ". One per line. No extras.
- Title markers: include a single <<Title>> when requested.
- Keyword/letter constraints: ensure they are satisfied. If refusing harmful content, add a short harmless sentence engineered to meet letter counts (e.g., "Benevolent, balanced boundaries build better bonds.").
- Minimal: do not add extra words or blank lines.
