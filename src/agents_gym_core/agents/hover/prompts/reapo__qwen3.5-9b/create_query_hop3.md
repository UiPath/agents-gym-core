Create Query Hop 3 Prompt (Round 2 - Final Synthesis)
=====================================================
Given the fields `claim`, `summary_1`, `summary_2`, produce the fields `query`.

Your objective: Generate a FINAL comprehensive query that retrieves ALL missing key entities needed to verify the complete multi-hop evidence chain.

FAILURE TAXONOMY AND CORRECTIVE STRATEGIES:
--------------------------------------------------------------------------------
Category 1: FINAL CHAIN ELEMENTS NOT EXTRACTED
  - Problem: After 2 hops, still missing entities from expected set
  - Example: Expected 3 entities but only found 2
  - STRATEGY: Perform a MAPPING audit between expected output and extracted entities
    - Expected: [Entity A], [Entity B], [Entity C]
    - Found in Hop 0: [matches A?] [matches B?] [matches C?]
    - Found in Hop 1: [which are new?]
    - MISSING: [list entity names that weren't found]

Category 2: COMPREHENSIVE FINAL QUERY FORMULATION
  - Problem: Queries that don't leverage full chain information
  - Strategy: Include ALL connected entities in final query to get everything at once
  - Template: "Identify: [Entity A], [Entity B], [Entity C] with relationships"
  - Template: "Document: [A] created by [B], connected to [C]"

FINAL QUERY FORMULATION INSTRUCTIONS:
--------------------------------------------------------------------------------
1. Create a mapping table:

   EXPECTED ENTITIES (from expected output or claim):
   - [Name 1]: [role, e.g., "the film"]
   - [Name 2]: [role, e.g., "the author"]
   - [Name 3]: [role, e.g., "the studio"]
   
   FOUND IN summary_1:
   - ✓ [name found]: [which role does it match?]
   - ✗ [missing]: [which role?]
   
   FOUND IN summary_2:
   - ✓ [name found]: [which role?]
   - ✗ [still missing]: [which role?]

2. List MISSING ENTITIES by role:
   - Missing [role 1]: [suggest entity name from claim context]
   - Missing [role 2]: etc.

3. Construct comprehensive query:
   - Best approach for multi-entity retrieval:
   - "Retrieve: [Role 1 entity], [Role 2 entity], [Role 3 entity]"
   - OR: "What are the [type] that connects: [Entity 1] and [Entity 2]?"
   - OR: "[Central entity] is linked to [entity a], [entity b], and [entity c]"

4. DO NOT:
   - Repeat the same query from Hop 1
   - Forget any entity from the claim
   - Only query for one entity when multiple are missing

OUTPUT:
A comprehensive query that will retrieve ALL missing entities needed to complete the chain from Hop 1+.
