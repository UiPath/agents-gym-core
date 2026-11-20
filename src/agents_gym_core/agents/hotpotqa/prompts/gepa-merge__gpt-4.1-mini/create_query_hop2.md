Task: Given a natural language `question` and a `summary_1` that contains partial information relevant to answering the question, generate an optimized `query` to be used as the **second hop query** in a multi-hop retrieval system.

Context:
- The retrieval system works in multiple hops:
  - **First hop:** Uses the original question to retrieve an initial set of documents.
  - **Second hop:** Your generated `query` is used to retrieve additional relevant documents **that were not retrieved in the first hop**, crucial for answering the question fully.

Input:
- `question`: The original complex question posed by the user.
- `summary_1`: A summary of information extracted from documents retrieved in the first hop related to the question. This summary partially addresses the question but may miss some entities or facts necessary for the final answer.

Output:
- `query`: A refined and targeted natural language query designed specifically to:
  1. Identify and retrieve **missing relevant documents** or entities not found in the first hop.
  2. Connect known facts from `summary_1` with gaps or alternative angles needed to complete the answer.
  3. Target documents related to entities or attributes that have been overlooked or underrepresented in the first-hop retrieval.

Guidelines and Important Details:
- Your goal is **not** to re-query what was already retrieved, but to help uncover the **remaining necessary documents** or information.
- Analyze what documents or entities were covered in the first hop (as hinted by `summary_1`) and identify which relevant ones are missing.
- Use clues and connections extracted from `summary_1` to formulate the query. For example:
  - If part of the answer involves comparing two entities and one was already retrieved, the query should focus on retrieving information about the other entity.
  - If the summary confirms some facts but leaves another related entity or property ambiguous, craft a query that explicitly targets those ambiguous or missing aspects.
- Avoid broad or generic queries that merely rephrase the original question. Instead, be as specific as possible about what is missing or unknown after the first-hop retrieval.
- Consider possible alternative relations or linked topics that can bridge the information gap.
- You may ask about related entities, events, locations, or attributes that complement the known facts.
- The generated query should be concise yet sufficiently descriptive to retrieve the missing entity documents.
- Use natural language questions or phrases that are likely to surface relevant documents in a retrieval system.

Examples of strategy:
- When a question contrasts two entities and the summary confirms facts about only one, generate a query focusing on retrieving information about the other entity.
- When the summary states a negative or limiting fact (e.g., a stadium did not host certain teams), the query should explore other possibilities or related details not yet covered.
- When the original question includes multi-part dependencies, your query should isolate the missing link or fact necessary to complete the chain.

By following these principles, the queries you generate will improve the multi-hop retrieval system's ability to gather all relevant documents needed to answer complex questions accurately.
