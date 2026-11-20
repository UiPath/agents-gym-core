# Task: Summarize passages for multi-hop question reasoning

Given the fields `question`, `passages`, and optionally `context`, produce a structured summary of relevant factual details.

## Goal
Extract SPECIFIC facts that will help answer a multi-hop question by precisely connecting information across passages.

## Critical Extraction Strategy
1. **Identify the target**: What EXACT entity or concept is the question asking about?
   - If it's a name, capture the FULL name with middle initials, quotes, street addresses
   - If it's a date, capture EXACTLY as written (day/month/year or year alone)
   - If it's a number/count, capture the EXACT value
2. **Search queued for target**: When passages appear, immediately check:
   - Does this contain the answer?
   - Does this bridge to information needed for the answer?
   - Does this provide context linking to the target entity?
3. **Extract complete entities**: Never truncate key information:
   - Names: "Varazdat Samuel \"Varaz\" Samuelian" not "Varaz Samuelian"
   - Organizations: "The New York Times" not "New York Times"
   - Locations: Use the EXACT location mentioned - don't ADD extra info
   - Example: "Braunschweig, Lower Saxony" not "Braunschweig, Lower Saxony, Germany" if only first two parts are in sources
   - Book titles: Include quotes if in source
4. **Bridge multi-hop**: Explicitly note connections:
   - "Passage [1] mentions that X is the author of Y"
   - "Passage [2] mentions Z which relates to Y from Passage [1]"
5. **Keep factually precise**: Only state what's explicitly in passages
   - DO NOT infer or ADD information not stated
   - DO NOT add countries, provinces, or details not explicitly mentioned
6. **Note ALL relevant dates/numbers/identifiers**: They're often the answer

## Structured Output Format
Organize summary by:
- **TARGET ENTITY**: Name, title, full designation EXACTLY as written
- **ENTITY DETAILS**: Full description matching source character-for-character
- **RELATIONS**: How entity connects to others
- **CRITICAL FACTS**: Dates, numbers, identifiers, locations - EXACT formatting
- **EVIDENCE**: Passage references supporting each claim

## Key Extraction Checklist
- [ ] Is the full name/term captured exactly?
- [ ] Are all dates in exact format?
- [ ] Are all numbers/IDs captured precisely?
- [ ] Is the exact spelling extracted?
- [ ] Is there a clear path to the answer through these facts?
- [ ] Are quotes/punctuation included where present in source?
- [ ] Did I NOT add extra information (countries, extra details)?

## Critical Exclusion
- DO NOT add country/region names if not explicitly in passage
- DO NOT add explanatory phrases when raw fact suffices

[[ ## reasoning ## ]]
{reasoning}
[[ ## summary ## ]]
{summary}
[[ ## completed ## ]]
