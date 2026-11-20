Summarize 1 Prompt (Round 2)
============================================
Given the fields `claim` and `passages`, produce the fields `summary`.

Your objective: Extract ALL key entities mentioned in the claim and ALL relevant facts from passages that help verify each entity.

FAILURE TAXONOMY AND CORRECTIVE STRATEGIES:
--------------------------------------------------------------------------------
Category 1: INCOMPLETE ENTITY EXTRACTION (CRITICAL)
  - Problem: Agent extracts SOME entities but misses others
  - Example: Expected ['john wain', 'young shoulders', 'ödön von horváth'] but only extracted ['young shoulders', 'ödön von horváth']
  - SUB-CAUSE: When passage says "Young Shoulders is a 1982 novel by John Wain", agent only extracts "Young Shoulders" not "John Wain"
  - SUB-CAUSE: When passage says "producer by Worldview Entertainment", agent only extracts "Worldview Entertainment" (GOOD!) but sometimes misses the film title
  - STRATEGY: EXTRACT EVERY PROPER NOUN PHRASE that could be an entity:
    - Person names: Any name after "by [PERSON]", "written by [PERSON]", "directed by [PERSON]"
    - Work titles: Any text following "novel by", "film by", "film called", "book ___", "title ___"
    - Organization names: Any name after "produced by", "produced worldview entertainment"
  - RULE: If passage says "X is a [noun] by Y" where Y is a person/organization, EXTRACT BOTH X AND Y

Category 2: Entity Disambiguation Errors
  - Problem: Confusing similar names or related entities
  - Example: Mix-up between "Garry Marshall" and "Charlotte Riefenstahl"
  - Strategy: Always quote the EXACT entity name as it appears. Do not paraphrase. Use FULL first name + LAST NAME or full title.

Category 3: Missing Claims to Entity Linkage
  - Problem: Not connecting entities BACK to what they represent in the claim
  - Strategy: For each extracted entity, ask "What does this entity represent in the claim?" and annotate:
    Entity: [name] → Represents: [role from claim, e.g., "The film", "The author", "The studio"]

RESPONSE REQUIREMENTS:
--------------------------------------------------------------------------------
1. List EVERY extracted entity with:
   - Entity name (EXACT from text)
   - What role it plays in the claim (film, author, company, etc.)
   - Source passage number
   
2. Verify each part of the claim against passages
   - Claim part: [quote claim segment]
   - Status: [confirmed / refuted / not in passages]
   - Evidence: [passage text or "not found"]

3. PREPARE FOR HOP 1: List all entities that need verification next
   - Format: "Missing: [entity name that needs retrieval]"

OUTPUT FORMAT:
- Extracted entities (with roles):
  1. [Entity Name as exact text] - Role: [in claim context] - From passage [N]
  2. [Entity Name as exact text] - Role: [...] - From passage [N]
- Verification:
  - Claim part "[part of claim]": [verified / needs more info]
- Hop 1 priority: [focus on connecting the missing piece]
