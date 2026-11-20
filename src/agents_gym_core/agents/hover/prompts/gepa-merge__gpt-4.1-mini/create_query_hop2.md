Task Description:  
You are given two fields: `claim` and `summary_1`. Your goal is to produce a field called `query` — a concise set of relevant, fact-checking questions or search style queries that can be used to retrieve evidence documents verifying or refuting the claim based on information contained or inferred from the summary.

Input Format:  
- `claim`: A factual statement potentially containing multiple fact assertions about people, events, attributes, titles, dates, roles, or relationships.  
- `summary_1`: A short paragraph summarizing factual information related to the claim, often clarifying or correcting some parts of the claim.

Output Format:  
- `query`: One or more specific, well-phrased questions or keyword queries that directly target the key factual discrepancies or verifications raised by the claim in light of the summary.

Detailed Instructions:  
1. **Extract key factual elements from the claim** — names, dates, titles, roles, events, or relationships explicitly or implicitly stated.  
2. **Contrast these facts with the summary to identify points of agreement, contradiction, or ambiguity.**  
3. **Formulate fact-checking queries that are:**
   - Tightly focused on the core factual issues raised by the claim and addressed or contradicted by the summary.  
   - Include named entities, dates, roles, or other domain-specific identifiers directly mentioned in both claim and summary to improve retrieval effectiveness.  
   - When relevant, break complex claims into multiple queries ensuring each fact is verifiable separately.  
4. **When relevant details appear only in the summary but are hinted at or missing from the claim (e.g., specific titles, roles, or names), include these in the queries to enable retrieval of key evidence.**  
5. **Use a clear, natural question format or targeted keyword phrases that could serve well as search queries.**  
6. **Avoid overly broad or generic queries; precision improves evidence retrieval quality.**  
7. Optionally, you may provide brief reasoning internally (not required in output) to ensure the queries cover all claim aspects and reflect the summary insights.

Examples of typical query components include:  
- Correct dates of events or deaths.  
- Confirmation of a person’s role or association with a known work.  
- Verification of relationships or allegations.  
- Details about specific cultural or domain elements (e.g., operas' structure, song directors).  
- Clarifying entity attributes or classifications (e.g., ethnicity, nationality).

By following these instructions, you will produce queries that are both comprehensive and targeted, maximizing the chance of retrieving relevant factual evidence relevant to the claim verification task.
