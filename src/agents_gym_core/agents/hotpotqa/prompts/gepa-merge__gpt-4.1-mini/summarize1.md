You are given two input fields: `question` and `passages`. Your task is to generate a concise, well-structured `summary` based on the provided passages that directly addresses the question.

This summary serves as the first-hop summarization in a multi-hop QA system and must accomplish two key objectives:

1. **Facilitate effective follow-up querying:** Include crucial entities, their relationships, and bridging facts that might currently be missing but are needed to generate precise follow-up queries in subsequent hops.

2. **Provide a solid foundation for final answer generation:** Capture definitive, relevant facts that are clearly supported by the retrieved passages and help downstream components arrive at the correct answer.

**Detailed Guidance:**

- **Focus on synthesis over compression:** Don't merely shorten the information. Instead, integrate relevant facts from all the passages to form a coherent, informative, and focused summary.

- **Include key entities with short factual context:** When passages mention people, organizations, shows, or other entities critical to answering the question, include their key attributes (e.g., titles, roles, dates, relationships) that are explicit or strongly implied in the passages.

- **Signal missing but necessary information:** If the evidence hints at connections or facts not fully covered by the current passages but required to resolve the question, highlight them indirectly by including related entities or possible bridging concepts. This helps ensure follow-up retrieval targets the right knowledge gaps.

- **Use exact or near-verbatim factual representations:** Summaries should contain accurate details like full names, dates, titles, roles, and relationships drawn faithfully from the passages.

- **Prefer clarity and precision:** The summary should make the key facts and their relevance to the question unambiguous, enabling later modules to focus on missing details efficiently.

- **Avoid speculation:** Do not inject external knowledge or guess answers if the passages do not provide them; instead, represent the known facts and highlight what is missing.

- **Maintain logical structure:** Organize the summary so that each important entity or concept is introduced with context that reveals its connection to the question.

**Example format elements:**  
- Entity Name | Brief description with relevant attributes.  
- Relationships or roles clearly linked to the question.  
- Dates, titles, or other factual data related to the question's core concepts.

**In essence:** Your summary is a synthesized distillation of retrieved evidence that bridges the gap toward a full answer by encapsulating all relevant factual clues and signaling gaps, thereby enabling effective subsequent retrieval and reasoning steps.
