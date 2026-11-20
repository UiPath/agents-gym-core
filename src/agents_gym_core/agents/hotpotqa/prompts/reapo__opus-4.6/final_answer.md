Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`.

CRITICAL RULES FOR THE ANSWER:
- Your answer must be as SHORT and CONCISE as possible — typically just a name, number, date, yes/no, or a brief noun phrase.
- Do NOT write a full sentence. Do NOT repeat the question. Do NOT add explanations or context.
- For yes/no questions, answer ONLY "yes" or "no" (lowercase).
- For "who" questions, give ONLY the person's name.
- For "what" questions, give ONLY the thing/entity name.
- For "where" questions, give ONLY the place name.
- For "when" questions, give ONLY the date or year.
- For "how many" questions, give ONLY the number.
- For comparison questions (e.g., "which is taller"), give ONLY the name of the entity.
- Use the EXACT phrasing, names, numbers, and dates as they appear in the summaries — do not paraphrase or reformat.
- If the answer is a number with commas (e.g., "156,257"), keep the exact format.

ENTITY TYPE RULES:
- Read the question carefully to determine WHAT TYPE of entity is being asked for (person, place, film, occupation, etc.) and answer with ONLY that type.
- If the question asks "what film/movie", answer with the film title only, NOT the actor or director.
- If the question asks "what occupation/profession", answer with the singular form of the occupation (e.g., "professional wrestler" not "professional wrestlers").
- Match the grammatical number of the question: singular question → singular answer.

NAMING RULES:
- For people: if the summaries provide a full name with middle names or nicknames in quotes, use that full form. But do NOT add titles or descriptions that aren't part of the name (e.g., "Ernest II" not "Ernest II, Duke of Saxe-Coburg and Gotha").
- For organizations: use the shortest recognizable name (e.g., "University of Missouri" not "University of Missouri Tigers football team"; "PATH" not "PATH rapid transit system").
- For sports records: use compact format with en-dash (e.g., "68–86" not "68 wins and 86 losses").
- For dates: include the full date as given in the summaries (e.g., "May 15, 1940" not just "1940"; "August 2, 1973" not just "August 2").
- Strip unnecessary qualifiers that aren't part of the core answer.
