Summarize 2 Prompt (Round 2)
=============================================
Given the fields `claim`, `context`, `passages`, produce the fields `summary`.

Your objective: Extract ALL exact document titles from passages that answer the query and complete the entity chain.

FAILURE TAXONOMY AND CORRECTIVE STRATEGIES:
--------------------------------------------------------------------------------
Category 1: INCOMPLETE ENTITY EXTRACTION FROM SECOND HOP
  - Problem: Agents find some entities but miss others from expected set of 3
  - Example: Expected ['keurig green mountain', 'microchip technology', 'pickit'] but got extra coffee docs
  - SUB-CAUSE: Agent extracts only the first few documents that mention the target, misses others
  - STRATEGY: Scan ALL passages and extract EVERY entity that matches the query intent
    - If query was about "[X]", extract EVERY document that contains [X]
    - If query was about "[connection]", extract relevant participators
    - Don't just extract first mention - scan full passage set

Category 2: Missing Entity in Retrieval (Recall Failure)
  - Problem: Some entities appear in retrieved passages but aren't extracted
  - Example: "John Wain" appears in passage but not extracted  
  - SUB-CAUSE: Agent focuses on query-target entity ("Young Shoulders") but ignores connected entities
  - STRATEGY: For EACH passage, extract:
    1. The primary entity the passage is about (document title)
    2. EVERY person name, company name, work title mentioned IN THAT PASSAGE
    3. They may not be the PRIMARY entity but they're still RELEVANT
  - RULE: Romance all documents! Not just the first one

Category 3: Over-Retrieval (Precision Failure)
  - Problem: Agent retrieves too many unrelated documents
  - Example: 19+ documents for a query expecting exactly 3
  - Strategy: This will be handled by BM25; my job is to extract ONLY what's ON TOPIC from retrieved docs
  - Rule: If the passage is clearly related to the query, extract its title. If not, skip it.

ENTITY EXTRACTION RULES:
--------------------------------------------------------------------------------
1. For EACH passage, identify:
   - Primary entity: What is the document/TITLE this passage is about?
   - Connected entities: Any persons, organizations, titles mentioned that relate to the claim
   
2. Extract format for each:
   - [Entity as exact phrase from text]
   
3. If the passage mentions multiple entities (e.g., "film by John Wain"), extract ALL of them as separate entries with different roles

OUTPUT REQUIREMENTS:
- List each extracted entity with its exact name and role
- Format: "[Entity name]" - Role: [what it is, e.g., "The film", "The author", "The studio"]
- Be exhaustive: Extract EVERY named entity that could help complete the evidence chain
