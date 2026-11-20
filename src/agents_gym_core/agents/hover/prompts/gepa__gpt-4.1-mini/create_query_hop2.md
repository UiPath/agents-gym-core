Task Description:

You are given two inputs:
- `claim`: a factual or composite statement that requires verification or investigation.
- `summary_1`: a detailed textual summary that discusses the claim and provides supportive or contradicting evidence from various sources, passages, or contexts.

Your goal is to generate a concise, precise, and targeted `query` string designed to retrieve or identify the most relevant evidence from a knowledge base or document collection that will help verify or refute the claim. This query will be used in information retrieval to find facts that confirm or disprove the claim's key components.

---

Detailed Instruction for Query Generation:

1. **Purpose of the Query:**
   - The query should be purpose-built to locate specific factual evidence or sources that directly confirm or contradict the claim.
   - Avoid general or vague terms that do not focus retrieval on the claim’s critical elements.

2. **Incorporate Critical Keywords and Entities:**
   - Extract and explicitly include all named entities (people, places, organizations), key dates, titles (e.g., of festivals, films, books), and other proper nouns mentioned both in the claim and prominently in the summary.
   - Include specific contextual qualifiers provided in the summary that clarify or link elements of the claim (e.g., birth dates, founder names, film titles, actor/director names, location names).
   - Add related or alternative names, titles, or aliases mentioned that are relevant to distinguishing facts and avoiding ambiguity.
   - When the summary mentions multiple related items or festivals, include their names explicitly to potentially retrieve comparative evidence.

3. **Resolve Ambiguity or Confusion Highlighted by the Summary:**
   - If the summary clarifies confusion or incorrect attribution present in the claim, the query must explicitly ask about the distinct entities or clarify relationships.  
     *Example:* If two characters or creators are confused, include queries for both alongside their roles and show titles.
   - Include both the potentially confused entities with their full identifying contexts (such as their professions, roles, or associated works).

4. **Connect Claim Components Logically:**
   - The query should combine crucial claim elements in a way that the retrieved evidence will cover all significant parts simultaneously or show their interrelation.
   - For composite claims (e.g., involving a birth date, a festival founder, and a film), include all these components coherently in the same query or in multiple nested queries.

5. **Be Precise and Focused:**
   - Keep the query focused on the core factual points needed to verify or refute the claim.
   - Use keyword or phrase fragments separated by semicolons or commas — full sentences are not necessary unless clarity demands it.
   - Do not introduce extra assumptions, unsupported facts, or external knowledge outside of the claim and summary.

6. **Include Multiple-Part Queries When Needed:**
   - If the claim is complex and involves several distinct entities or facts that don’t easily combine in a single query, formulate multiple targeted queries to cover each aspect.
   - This helps separate retrieval tasks and avoids missing partial evidence.

7. **Incorporate Evidence Missing From Previous Queries (According to Feedback):**
   - When prior outputs reveal missing important entities or keywords (e.g., names like "Václav Havel," "Garry Marshall," or terms like “Summer Shakespeare Festival”), ensure these are integrated explicitly.
   - Always revisit the summary’s mentions of such critical entities or qualifiers even if not originally included in the claim, as they may be necessary for complete evidence retrieval.

---

Summary of Best Practices From Examples and Feedback:

- Always link proper nouns distinctly with qualifying details (e.g., "Los Angeles Free Shakespeare Festival" and "Terrence Scammell founder") rather than generic terms such as “Summer Shakespeare Festival” alone.
- Specify birth dates and relationships explicitly to retrieve biographical comparisons.
- When a film is involved, include both the film title and key production personnel (e.g., director, cinematographer) to pinpoint evidence.
- If a summary reveals no information about some components, generate queries asking specifically for basic unprovided facts (e.g., identity and birth date of a filmmaker).
- Combine claim entities and summary clarifications for a more comprehensive query, avoiding missing keywords that blocked evidence retrieval previously.

---

Example Query Construction Format:

`[Entity or keyword 1] [relationship or qualifier]; [Entity or keyword 2] [relationship or qualifier]; [Entity or keyword 3] ...`

*E.g.*  
`Penelope Fitzgerald birthdate; Terrence Scammell founder Los Angeles Free Shakespeare Festival birthdate; Freewill Shakespeare Festival founder birthdate; Shakespeare in Styria founders; Shakespeare on the Saskatchewan founders`

---

This strategy will maximize the relevancy and comprehensiveness of the queries, enabling retrieval of precise evidence needed to confirm or disprove complex claims from the provided summaries.
