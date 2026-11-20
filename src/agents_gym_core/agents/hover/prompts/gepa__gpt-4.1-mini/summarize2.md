Given three input fields: `claim`, `context`, and `passages`, your task is to generate a concise, precise, and fully evidence-grounded `summary` that explicitly states whether the claim is supported, refuted, or not confirmed based solely on the information provided in the `context` and `passages`. 

Key requirements for the `summary`:  
- Clearly articulate the truthfulness status of the claim (e.g., supported, refuted, or not confirmed).  
- Explicitly link relevant named entities, key facts, dates, works (e.g., films, albums, universities), relationships, or events referenced in the `claim` to corresponding or contradictory evidence in the `context` and `passages`.  
- Include domain-specific factual details—such as specific names, titles, alternate names, affiliations, dates, university campuses, sports leagues, label ownership, or career credentials—that directly connect the claim to the referenced evidence.  
- Avoid vague or generic statements; instead, embed detailed facts that show the direct relationship between elements in the claim and the evidence.  
- Ensure the summary is succinct but comprehensive enough to cover all critical evidence facets required for precise query formulation or key evidence extraction in downstream verification tasks.

Detailed instructions and strategies for generating the summary:  
- Carefully cross-check every element in the claim against the facts in both the `context` and all provided `passages` to verify consistency or identify contradictions.  
- When the claim references persons (e.g., musicians, professors), works (e.g., films, songs, albums, biographies), affiliations (e.g., labels, leagues, university systems), or terminology with variants or multiple adaptations, explicitly mention these relevant variants or relationships if they appear in the evidence to strengthen clarity and reasoning. For example:  
  - If the claim references a film associated with a person, mention related titles or adaptations if referenced in the evidence.  
  - If a claim involves collaborations or signings (e.g., a band signed by a label owner who is also a musician), identify both the signer and their roles explicitly.  
  - When discussing university systems or campuses, specify the exact institutions tied to individuals and whether any combination (such as combined acreage or enrollment) is supported by evidence.  
- Highlight any subtle or implicit evidence that forms connections between the claim and passages/context, such as: relevant university names, official positions held, definitions of entities (e.g., defining what a university system includes), or league names for sports teams.  
- If certain elements in the claim have no supporting or contradicting evidence in the input, explicitly state the lack of evidence to confirm or refute that part of the claim.  
- Do not assume any information outside the provided context and passages, even if it might be widely known externally.  
- Reflect facts in a manner that assists downstream tasks like generating focused queries or extracting pinpoint evidence by explicitly naming and connecting entities and relationships.  
- Use exact terms and names from the input texts (e.g., “West Cheshire League Division Two,” “Posthuman Records,” “University of Florida,” “Marilyn Manson,” “Joe Lynn Turner,” or “Godhead (band)”) to ensure traceability and specificity.

Overall, your summary should serve as a tightly reasoned, factually comprehensive bridge between the claim and the evidence, illuminating key connections or conflicts, including explicit references to crucial evidence and domain-specific details essential for precise verification and evidence retrieval workflows.
