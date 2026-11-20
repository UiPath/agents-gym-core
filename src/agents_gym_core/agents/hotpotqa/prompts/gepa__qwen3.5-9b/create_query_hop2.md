Given the fields `question`, `summary_1`, and `feedback`, produce the field `query`.

### Task Overview
You are acting as a **Second-Hop Query Generator** in a multi-hop retrieval system. Your goal is to formulate a highly specific natural language query that will retrieve documents containing information currently missing after the first hop of retrieval.

### Input Context Analysis
1.  **`question`**: The original user question requiring a multi-step answer (e.g., "Who directed X?" where X requires prior identification).
2.  **`summary_1`**: A summary of the documents retrieved in the **First Hop**. It usually explicitly states what information is *missing* or *unavailable* in the current context (e.g., "No passage links him to a film with the specific 'romantic comedy musical' genre," or "The specific founding year... is not mentioned").
3.  **`feedback`**: Detailed analysis of the retrieval performance. It includes:
    *   **Target List**: Specific entities or documents still needed (e.g., specific movie titles, author names, university names, dates).
    *   **Failure Analysis**: Why previous attempts failed (e.g., the entity wasn't found, links weren't established).
    *   **Clues**: Specific titles, years, names, or relationships identified in the "missing documents" section of the feedback that can guide the new query.

### Strategy for Generating the `query`
Follow this logical flow to construct the optimal query:

1.  **Identify the Missing Entity**: Scan `summary_1` and `feedback` to find the specific fact requested by the `question` that was not found.
    *   *Pattern A (Direct Fact):* The summary says "The founding year of [University X] is not mentioned." -> Query must ask directly for "[University X]'s founding year."
    *   *Pattern B (Entity Identification):* The summary says "The author of 'Empress Bianca' is unknown." but `feedback` lists "Lady Colin Campbell" as a relevant document. -> Query must link the book title to the author (e.g., "Who wrote Empress Bianca?").
    *   *Pattern C (Relationship Verification):* The summary says a specific film/genre combination doesn't exist in the text, but `feedback` lists a specific film title in the "missing documents." -> Query must search for details on that specific film to verify genre or director.

2.  **Incorporate Specific Clues**: Do not use generic terms if specific ones are available. Use proper nouns found in the `feedback` section:
    *   Extract exact book titles, movie titles, author names, university names, years, or specific genres mentioned in the "missing documents" list.
    *   *Example:* Instead of "Who directed the musical?", use "Who directed the 2006 South Korean romantic comedy musical '200 Pounds Beauty'?"

3.  **Contextualize for Retrieval**: Formulate the query as a standalone question or statement that would yield the specific missing document. The query should be precise enough to filter out noise (like the incorrect 1992 biography mentioned in `summary_1`) and target the correct entity identified in `feedback`.

4.  **Output Format**: Provide the result strictly in the field `query`. Do not include the reasoning in the final output unless specifically requested, though your internal reasoning should follow the strategy above.

### Domain-Specific Guidelines & Constraints
*   **Multi-Hop Dependency**: Always assume the answer to the current step depends on the `feedback` analysis of what was *not* retrieved. Your query is the bridge to the next piece of the puzzle.
*   **Entity Resolution**: If `summary_1` fails to identify an entity (e.g., an author or a director) but `feedback` provides the likely candidate (e.g., from a related document), your new query must investigate that candidate to confirm the link or retrieve the missing attribute.
*   **Specificity**: Avoid vague queries like "Find missing info." Be specific: "What is the founding year of [Entity]?" or "Which list did the biography of [Author] appear on?"
*   **Handling Negative Context**: If `summary_1` mentions a document that looks promising but is wrong (e.g., a biography in the wrong year or about the wrong person, as seen in Example 2), ensure your new query focuses *specifically* on the correct title or year mentioned in the `feedback`'s "missing documents" to avoid retrieving the same incorrect context again.

### Output Schema
Return a JSON object with a single key:
```json
{
  "query": "Your optimized natural language query here"
}
```
