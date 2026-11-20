Task Description:

You are given three fields as input: `question`, `summary_1`, and `summary_2`. Your goal is to produce an `answer` field that directly and explicitly responds to the question using the information from the two summaries, enhanced by your authoritative domain knowledge when needed.

Input Format:

- `question`: A natural language question that may require a fact, definition, name, date, yes/no response, or other specific information.
- `summary_1` and `summary_2`: Two independently generated summaries or snippets containing information related to the question. They may vary in completeness, accuracy, and specificity.

Requirements and Approach:

1. **Understand the question precisely.** Determine exactly what is asked—whether a name, a specific fact, a date, or a yes/no answer.

2. **Compare both summaries.** Analyze the content of `summary_1` and `summary_2`:
   - If they agree and directly answer the question, use this as primary evidence.
   - If one summary provides a fact that the other does not mention, carefully evaluate its plausibility.
   - If the summaries conflict, use domain expertise and authoritative knowledge to resolve or explicitly state uncertainty.

3. **Domain-specific factual verification and nuance:**
   - **Names and nicknames:** Provide only the specific nickname or name when asked, without extra phrasing. For example, when asked for the nickname of a person or entity, respond with the nickname alone, not a full sentence.
   - **Nationality and identity distinctions:** Use the most precise terms aligned with factual correctness and common usage (e.g., "English" vs. "British") based on domain knowledge.
   - **Dates and historical facts:** Verify dates or historical claims with domain knowledge to pick the correct fact, especially when there might be confusion between franchise start dates vs. event dates etc.
   - **Yes/no questions:** Prefer concise answers of "yes" or "no" only, unless the question demands elaboration.
   - **Types or categories:** If a question asks about the type or category (e.g., type of company), provide the most direct concise phrase without including adjectives like nationality unless asked explicitly.

4. **Answer conciseness and relevance:**
   - Provide a brief and direct answer to the question.
   - Avoid repeating or restating the question.
   - Avoid unnecessary context unless requested or needed for clarity.
   - Avoid constructing full sentences unless needed; for example, answers to nickname or yes/no questions should be as short and specific as possible.

5. **When authoritative knowledge supplements the summaries:**
   - If the summaries are incomplete or potentially inaccurate, incorporate trusted knowledge from your training about the topic to provide the correct and precise answer.
   - For example, when a summary gives a year that conflicts with known release dates or factual details, prefer the verified date.
   - When the summaries differ in style (one uses a formal phrase, another provides just the nickname), respond with the correct, clean answer format (e.g., just the nickname alone).

6. **Examples of correct reasoning and answers:**
   - Question: "What is the nickname of the 2005 Toyota Grand Prix of Long Beach Polesitter ?"
     - Correct answer: `the thrill from West Hill`
   - Question: "What type of company is Zipcar led by Scott Griffith from 2003-2013?"
     - Correct answer: `car-sharing company`
   - Question: "Who was the partner of British comic book artist, Henry Flint, that helped create Zombo?"
     - Correct answer: `Al Ewing`

Summary:

- Use both summaries as primary but not sole evidence.
- Reliably verify and contextualize facts using domain knowledge, especially for nationality, dates, nicknames, company types, and yes/no questions.
- Provide short, direct answers matching the specificity requested.
- Avoid unnecessary elaboration unless explicitly required.
- Explicitly resolve conflicts or ambiguity using your knowledge or state uncertainty when appropriate.

This approach ensures that answers are both accurate and concise, suitable for direct consumption or integration in knowledge bases or question-answering systems.
