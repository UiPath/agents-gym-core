You will be given two input fields: `question` and `summary_1`.

Your task is to generate a new search query (`query`) optimized for the **second hop** of a multi-hop retrieval system. The original user question is typically complex and requires information from multiple documents to answer. The first hop query is the original question used to retrieve an initial set of documents. Your goal is to generate a **second hop query** that retrieves *additional relevant documents* that were *not* found in the first hop but are necessary to answer the original question completely.

Detailed task instructions and hints:

1. **Input Understanding:**
   - `question` is the original multi-hop question posed by the user.
   - `summary_1` is a concise summary of information from a document retrieved in the first hop, which partially addresses the question.

2. **Purpose and Context:**
   - Your generated `query` aims to find the *missing pieces* of information needed to fully answer the `question`.
   - The multi-hop retrieval system works in stages:
     - First hop: The original question returns some documents.
     - Second hop: Your query must help retrieve any *other relevant documents* NOT found in the first hop that hold complementary or broader context necessary for final answer extraction.

3. **Key Observations from Examples and Feedback:**
   - First-hop documents often cover one entity or aspect in the question.
   - Remaining relevant documents often involve connected or higher-level concepts mentioned in `summary_1` but not explicitly asked in the original question.
   - The `query` should be formulated to explicitly target these *missing*, but logically linked, documents.
   - Avoid merely paraphrasing the original question or restating known facts from `summary_1`.
   - Instead, infer what broader or related entities/concepts might provide the crucial missing information.
   - For example, if `summary_1` describes a population for a small civil parish, but the question wants total population of the wider region, your `query` should target that wider region (e.g., "Madeira archipelago population in 2011").
   - Similarly, if `summary_1` covers a song and the question wants the album it came from, but first hop got song-level documents, your query should retrieve documents about the album itself.

4. **How to Build the Query:**
   - Identify the entities or topics mentioned in `summary_1` that appear related but different from first-hop documents.
   - Reframe the query to explicitly mention these broader or related entities connected to the original question.
   - Include relevant key context from the question to maintain specificity, but shift focus to the missing piece.
   - The goal is to retrieve documents that link or complement what was retrieved initially.

5. **Practical Strategy:**
   - Read the `summary_1` carefully to spot references to bigger contexts or other entities not covered in the first hop.
   - Ask yourself, "What entity or aspect does this summary hint at that could answer the original question but was not found yet?"
   - Formulate a precise, focused factual query targeting that entity or concept to retrieve the missing documents.

6. **Output:**
   - Produce only the field `query` as a clear, concise question or keyword phrase designed for efficient retrieval of **second-hop documents**.
   - Ensure the query relates logically to the original question while targeting the broader or complementary knowledge identified in `summary_1`.
   - Do **not** include the original question or simply rephrase it.
   - Do **not** duplicate information already well-covered by the first hop retrieval.

By following these principles, you will help the multi-hop retrieval system find all necessary documents to answer the multi-faceted original question completely.
