Task Description:

Given two fields in the input:  
- `claim`: a factual assertion or statement that needs to be evaluated or contextualized.  
- `passages`: a list of textual passages containing information that may be relevant to the claim.

Your task is to produce a `summary` field that:  
1. Concisely and clearly synthesizes relevant information from the provided passages that directly supports, refutes, or contextualizes the claim.  
2. Explicitly connects key named entities, dates, or facts mentioned in the claim to the matching evidence or lack thereof within the passages.  
3. Highlights the core relationship(s) relevant to the claim, such as people, films, studios, events, locations, or roles, ensuring that the summary makes explicit references to these entities and how they correspond (or do not) to the claim.  
4. Avoids vague or generic wording; instead, it should name crucial pieces of evidence (e.g., names of films, individuals, studios, events, or locations) to facilitate effective retrieval of supporting or contradicting documents.  
5. States clearly when the information provided does not support or contradict the claim, specifying which parts remain unsupported or unverified.  
6. Remains focused on accuracy and completeness to help downstream tasks (e.g., query generation or evidence retrieval) by including all meaningful connections to the claim's entities.

Additional domain-specific insights to consider from examples:  
- Film and actor names, release years, film studios (including their geolocation relative to known places), and roles (director, actor, contract player) are critical for verifying claims about cinema history or entertainment figures.  
- "Co-stardom" implies actors appearing together in the same movie; confirm or deny this relation explicitly if stated.  
- For music-related claims, distinguish between artists' real names and stage names clearly, and verify their associations with festivals or venues by name.  
- Locations (e.g., nightclubs, studios) should be precisely linked to events/festivals or persons where relevant.  
- When multiple people with similar attributes are mentioned (e.g., newcomers to a film), explicitly identify and mention their names when discussing claims about them.  
- Avoid conflation of entities or incorrect equivalence (e.g., mixing up artist identities).

Generalizable Strategy:  
- Carefully parse the claim into its constituent assertions and identify key named entities and relationships.  
- Scan the provided passages for explicit statements about these entities and relationships.  
- Restate whether the evidence supports or contradicts each element of the claim, naming specific passages or facts where appropriate.  
- Include any relevant but indirect information that clarifies the claim’s truth value or context, while noting the absence of explicit evidence if applicable.  
- Keep the summary balanced, factual, and useful as a seed for further query or evidence retrieval.

In summary, your output must be a factually accurate, entity-rich, and evidence-linked summary explaining how the passages relate to the claim, explicitly connecting the claim’s elements to the available evidence or lack thereof.
