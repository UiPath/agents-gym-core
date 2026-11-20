Additional validator corrections:
- If a repeat-then-answer instruction is present and the response lacks the verbatim repetition first, insert it now using the exact original request text (excluding the meta line). Place it at the very beginning.
- Remove any labels ("A:", "Answer:") before the answer unless explicitly requested.
- For MCQs, if a letter was output, replace with the value only unless the prompt explicitly asks for the letter.
- Re-check end-phrase, bullet counts, titles, and keyword/letter constraints. If letter-frequency constraints exist and are unmet, append a short harmless sentence engineered to satisfy the count.
