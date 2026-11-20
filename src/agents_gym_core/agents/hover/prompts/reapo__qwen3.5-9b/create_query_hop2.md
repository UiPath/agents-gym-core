Create Query Hop 2 Prompt (Round 2)
====================================
Given the fields `claim`, `summary_1`, produce the fields `query`.

Your objective: Generate a query to retrieve the MISSING CRITICAL ENTITY IN the multi-hop chain.

FAILURE TAXONOMY AND CORRECTIVE STRATEGIES:
--------------------------------------------------------------------------------
Category 1: Entity Name vs Generic Topic Query
  - Problem: Querying "the film maker" instead of "John Wain"
  - Strategy: ONLY use exact entity NAMES from summary_1, never roles/descriptions

Category 2: Lost Chain Elements
  - Problem: Losing track of which entity in the claim chain is still missing
  - Strategy: Reconstruct the full claim chain:
    Claim: "[Entity A] [relation] [Entity B] [relation] [Entity C]"
    - Check which of A, B, C are confirmed in summary_1
    - If Entity B is missing but Entity A is known, query for Entity B
    - If ONLY missing the connecting entity (like a person who linked two films), query specifically for that entity

QUERY FORMULATION RULES:
--------------------------------------------------------------------------------
1. Re-read claim and identify the chain structure:
   Template: "[A]: [description], [connection] [B]: [description]"
   
2. Review what summary_1 confirmed:
   - Entities verified
   - What's still unknown

3. Extract the EXACT NAME of what's missing (from claim/summary_1)
   - BAD: "Query for the author"
   - GOOD: "Query for John Wain"

4. If missing a CONNECTION between known entities:
   - Template: "What [Person/Organization] is [artist/director/writer/star] of [Work]?"
   - Template: "Who [role] is [relation] between [Entity A] and [Entity B]?"

5. If missing a link in a telescope chain (A relates to B relates to C):
   - Use the known end point and relationship
   - Template: "What [B] is associated with [A] through [relationship]?"

OUTPUT:
A query using EXACT entity names from the claim/summary_1 to find the missing link(s).
