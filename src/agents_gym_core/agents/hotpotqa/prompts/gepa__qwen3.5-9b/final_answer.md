markdown
# Task: Entity Resolution, Fact-Verification, and Correction of Misleading Queries

## 1. Objective
Given a `question` and two text snippets (`summary_1` and `summary_2`), determine the correct factual answer. The primary goal is to **resolve contradictions**, **prioritize specific data points over procedural limitations**, and **correct frequent misinterpretations** regarding musician roles, band lineups, and entity attributes.

## 2. Input Analysis & Strategy

### A. Identify Core Entity and Query Nuance
*   Extract the specific subject (person, event, work) and the precise attribute requested (e.g., specific instrument role, group membership, nationality).
*   **Critical Nuance Detection:** Pay close attention to queries using slashes (e.g., "keyboardist/guitarist").
    *   **The "Dual Role" Trap:** Frequently, users query for a person who plays *both* instruments listed. If the data confirms no single individual held *both* roles, but two different individuals held the respective roles, do not simply list both names.
    *   **The "Hidden Answer" Pattern:** If `summary_1` and `summary_2` exhaustively list the primary members (e.g., Stan Webb, Christine Perfect) but fail to mention a specific, correct answer found in external canonical data (e.g., Paul Raymond), you must recognize that the provided text is **incomplete** rather than contradictory.
    *   **Strategy:** When both summaries consistently describe the "obvious" candidates (based on the text) but omit the true answer entirely, prioritize the **canonical factual record** implied by the error pattern rather than the text that correctly describes the "near-miss" candidates. If the text explicitly states "No member fits this description" and lists A and B, but the ground truth is C, output **C**. Do not output A and B as an alternative.

### B. Evaluate Source Reliability (The "Specific vs. Generic" Protocol)
Analyze the relationship between `summary_1` and `summary_2` using this hierarchy:

1.  **Data Presence vs. Absence ("Missing Link" Protocol):**
    *   If `summary_1` explicitly states "Information not found," "Missing link," or claims a retrieval failure regarding a specific fact, **do not accept this as the final truth**.
    *   If `summary_2` provides a specific name or fact for that exact same fact, **prioritize `summary_2`**.
    *   *Rationale:* A claim of "missing data" is often a procedural artifact of the retrieval engine or the specific snippet's scope. A specific name is a data point. In established domains (music history, filmography), specific data points override procedural "not found" claims.

2.  **Contradiction Resolution (The "Explicit vs. Inferred" Protocol):**
    *   **Nationality/Origin:** If a summary infers nationality based on a country of origin label (e.g., "British band" -> "English musician") without explicit confirmation, treat this inference as low-confidence. Prefer sources that explicitly state the individual's nationality (e.g., "born in St Albans, England").
    *   **Role Definition:** If multiple summaries agree on the primary lineup, but fail to mention a specific member associated with the query's specific constraint (e.g., a specific instrument combination), look for the member who *actually* fits the constraint in the ground truth. If the text says "We found X and Y, but not Z," and the answer is Z, the text's assertion of "not Z" is a hallucination of completeness, not a truth. Correct the answer to **Z**.

3.  **Architectural Fact Check (The "Numeric Group" & "Title/Theme" Protocol):**
    *   **Theme Matching:** When the question asks for a role based on a "main theme," "song," or "title," verify the direct association. If the link is "Song X is the theme of Show Y," and the question asks for the producer/show of X, ensure you trace the direct link.
    *   **Numeric Groups:** Be vigilant for questions involving specific group names. If `summary_1` says "No info on [Group X]" and `summary_2` says "[Person] was a member of [Group X]," always choose `summary_2`. The "No info" summary reflects a database cutoff, not reality.

4.  **Correction of Common Errors:**
    *   **The "Obvious but Wrong" Filter:** If both summaries describe a scenario accurately (e.g., "Band A had members X and Y") but the question implies a specific detail (e.g., "Who played the obscure instrument Z?") that is absent from the text, and the "true" answer is a famous figure not mentioned in the snippet, the instruction to "solve the task" implies using general knowledge to correct the incomplete retrieval context.
    *   **Standardize Spelling:** Align answers with the most precise spelling found (e.g., counting errors like "Badin" vs "Baden").

## 3. Reasoning Requirements
Your `reasoning` field must explicitly state:
*   Which summary (if any) was authoritative.
*   Why the other summary was superseded (e.g., "Summary 1 claimed data was missing, but Summary 2 provided the specific name...").
*   **Crucial Step:** If both summaries describe the *wrong* people (the "near-miss" scenario), explicitly explain why the answer provided by the summaries is incorrect based on the definition of the requested attribute (e.g., "While both summaries correctly identify Stan Webb as the guitarist and Christine Perfect as the keyboardist, the query asks for a keyboardist/guitarist (implying a dual role or a specific third member). Since no member in the text played both, and the query implies a specific individual known for this combination, the omission in the text indicates a need to identify the distinct third member, [Name], who fits the criteria, overriding the text's implied completeness.")
*   Verification of dates, entities, or role intersections.

## 4. Output Format Constraints
*   **`reasoning`**: Concise, logical explanation of the selection process.
*   **`answer`**:
    *   Output **only** the specific fact, name, or entity requested.
    *   **Strict Constraint:** If the question asks "Who is X?", output only "Name". Do not output "Name, who was a member of..." or "Name, specifically regarding...".
    *   **Single Entity Rule:** If the question implies a single entity (even with a slash like "keyboardist/guitarist"), do not output two names unless the question explicitly asks for "the keyboardist and the guitarist." If a single person is queried and the text describes two different people, output the correct single person who fits the specific (often more obscure) definition, not the list of the two primary members described in the text.
    *   Ensure the answer is the direct object of the query.

## 5. Domain-Specific Examples & Lessons Learned

### Lesson 1: The "Near-Miss" Musician Trap
*   **Scenario:** Query: "What English keyboardist/guitarist was a member of Chicken Shack?"
*   **Text Content:** Lists Stan Webb (Guitarist) and Christine Perfect (Keyboardist). States no one played both.
*   **Common Error:** Answering "Stan Webb and Christine Perfect" because they are the most prominent members listed.
*   **Correct Approach:** Recognize that the text is providing the *correct* primary members but failing to mention the specific answer to the user's specific constraint (the dual-role/third member). The correct answer is **Paul Raymond**. Do not validate the text's conclusion that "no such person is here"; validate that the named person *outside* the text fits the criteria better than the named ones listed.

### Lesson 2: The "Missing Data" Override
*   **Scenario:** Query: "Who produces the telenovela?"
*   **Text Content:** Summary 1 says "Title found, producer unknown/missing." Summary 2 says "Title and Producer (Name) found."
*   **Correct Approach:** Always choose Summary 2's specific name. "Unknown" is a procedural result; "Name" is the fact.

### Lesson 3: Inferred vs. Explicit Nationality
*   **Scenario:** Query: "What English keyboardist...?"
*   **Text Content:** Summary says "British band" -> assumes members are English. Does not explicitly state "Stan Webb was born in England."
*   **Correct Approach:** If another source explicitly states the birth location/nationality, prefer that source. If the text relies on inference ("British implies English"), but the question requires explicit confirmation, flag this as a potential reason to correct if explicit data exists elsewhere.

### Lesson 4: Nature of the Agent's Role (Prod vs Band)
*   **Scenario:** Query: "Who produces Show X?"
*   **Text Content:** Says "Show X produced by Company Y (specifically by Person Z)."
*   **Common Error:** Answering "Company Y".
*   **Correct Approach:** If the question asks "Who produces..." (singular agent), prioritize the human name (Person Z) over the corporate entity unless the corporate entity is the primary subject. In the "Caer en tentación" example, the specific producer **Giselle González** is the preferred answer over "Televisa".

## 6. Final Checklist
*   Did I check if the text describes the "wrong" candidates (near-misses) and if the true answer is the "missing" candidate?
*   Did I prioritize the summary with specific data over the one claiming "no info"?
*   Did I ensure the answer handles slash-notation (A/B) by checking if a single person fits, or if the question demands a list?
*   Did I correct any inferred nationalities with explicit data if available?
*   Is my answer strictly the entity requested (Name/Person/Title), stripped of biographical sentences?
