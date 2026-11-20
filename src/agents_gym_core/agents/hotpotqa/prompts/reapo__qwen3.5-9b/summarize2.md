# Task: Summarize passages for multi-hop question reasoning (second pass)

Given the fields `question`, `context`, and `passages`, produce a refined summary incorporating the context and verifying facts from new passages.

## Goals
1. **Validate**: Check if context provides confirmation or correction
2. **Extract**: Capture any NEW facts from the new passages
3. **Synthesize**: Combine information that wasn't available in summary_1
4. **Connect**: Bridge gaps between summary_1 and new information

## Critical Focus Areas
- **Recall from summary_1**: Remember what was already known
- **New entity extraction**: Names, dates, numbers, locations
- **Relationship verification**: Does new info confirm or contradict?
- **Multi-hop bridge**: If this info connects to previous info, note the connection
- **Answer readiness**: After this summary, should the question be answerable?

## Output Requirements
- Be even more specific than summary_1
- Include any facts needed to answer the question directly
- Note if connection to summary_1 creates a complete knowledge chain
- Preserve exact wording for names, dates, and numbers

[[ ## reasoning ## ]]
{reasoning}
[[ ## summary ## ]]
{summary}
[[ ## completed ## ]]
