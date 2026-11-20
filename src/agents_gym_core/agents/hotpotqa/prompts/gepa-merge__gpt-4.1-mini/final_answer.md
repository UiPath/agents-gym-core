Task Description:

You will be given three input fields:  
- `question`: a natural language question specifically requesting a concise factual answer. The question will typically ask for a specific entity such as a person's name, a date, a place, a title, or a simple fact.  
- `summary_1`: a textual summary containing relevant facts, possibly with detailed context, that could answer the question.  
- `summary_2`: a second textual summary with overlapping or complementary relevant facts that also potentially answer the question.

Your objective is to produce a single output field:  
- `answer`: a concise, minimal, and direct factual answer extracted and inferred from the provided summaries.

Instructions and Best Practices:

1. **Precisely identify what the question asks for**  
   Determine the exact factual entity or value requested (e.g., name of person, title of movie, country, year, band name).

2. **Synthesize both summaries to validate and extract facts**  
   - Compare `summary_1` and `summary_2` for consistency on key facts relevant to the question.  
   - Use the summaries in combination to confirm or fill any missing details.  
   - If both summaries agree or nearly agree, trust the shared information.  

3. **Provide the exact minimal answer only**  
   - Supply only the essential answer without restating the question or adding explanatory context.  
   - For example, if the question asks for a person's name, provide the name alone (e.g., "Andre Dirrell").  
   - Do not include relational or descriptive words unless explicitly requested (e.g., do not say "Anthony Dirrell is the brother of Andre Dirrell" but simply "Andre Dirrell" if asked for the brother's name).  
   - In cases where the question expects a title, year, or place, provide just that fact.  
   - For choice questions (e.g., which band or film), supply only the correct choice name.  

4. **Ensure factual accuracy and clarity**  
   - The answer must be based solely on information found in the summaries.  
   - Avoid including personal opinions, unrelated facts, or extra details not directly answering the question.  

5. **Domain-specific considerations:**  
   - **Boxing domain:**  
     Understand that titles like "IBF interim super middleweight title" or "WBC super middleweight title" refer to professional boxing championships. Names often include birthdates or career details, which help verify identities but are usually not part of the answer unless asked. Relationships (e.g., brothers) are key but supply only the entity name when prompted.  
   
   - **Music domain:**  
     Bands' formation years, member roles, and member names distinguish groups. Provide just the band or member name as the answer. Roles or details (like "lead vocalist") are supplied only if the question directs you.  
   
   - **Entertainment domain:**  
     Years of birth, album titles, movie names, or director names identify time-based or identity-related answers. Provide just the year, name, or title as requested.  

6. **Reasoning and explanation should be optional and separate from your final output**  
   - When reasoning about your answer, you may highlight where both summaries agree or support the answer, mention confirming dates or facts, and summarize how you synthesized the information.  
   - The final output must only be the concise, factual answer field without reasoning or extra commentary.

7. **Common pitfalls and corrections:**  
   - Do not confuse the entity sought by the question with related entities mentioned in summaries (e.g., if the question asks for a movie title, do not answer with an actor's name even if both appear).  
   - If the question requests a movie or work title associated with an actor, answer with the title alone.  
   - Avoid quoting full sentences, pronouns, or relational phrases unless explicitly requested.

Input / Output Format:

Inputs:  
- `question` (string) — the factual question.  
- `summary_1` (string) — first relevant textual summary.  
- `summary_2` (string) — second relevant textual summary.

Output:  
- `answer` (string) — the minimal, concise factual answer strictly drawn from the summaries.

This approach ensures consistent, accurate extraction of direct factual answers from provided summaries across diverse domains like boxing, music, and general entertainment.
