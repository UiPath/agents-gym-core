# Task: Answer a multi-hop question using all summaries - WITH STRICT EXACT MATCH REQUIREMENTS

Given the fields `question`, `summary_1`, and `summary_2`, generate a FINAL ANSWER matching EXACTLY the expected output format.

## CRITICAL: Exact Match is Mandatory
The answer must match character-for-character what is expected. Even minor differences like capitalization, quotes, spacing, or singular/plural will cause failure.

## Answer Format Rules by Type
1. **For Yes/No Questions**: 
   - Output ONLY "yes" or "no" (lowercase, no extra spaces, no punctuation)
   - Example: "yes" - NOT "Yes", "YES", "Yes.", or "I think yes"

2. **For Names (People, Places, Organizations)**:
   - Include ALL parts: middle names, initials, quotes, epithets
   - Extract EXACT spelling as shown in passages
   - Example: "Varazdat Samuel \"Varaz\" Samuelian" with quotes
   - Example: "The United States Naval Academy" with "The" and full name
   - Example: "Aubrey O'Day" not "Aubrey Morgan O'Day" or "O'Day"
   - DO NOT add extra information (countries, states) if not explicitly in expected answer

3. **For Numbers**:
   - Output the NUMBER ONLY (no words, no units)
   - Preserve decimal places, commas if applicable
   - Year format: "2015", "2016" - just the digits
   - Counts: "255" not "255 individuals"
   - Note special numbers: "68–86" (en-dash), "68-86" (hyphen)

4. **For Challenge Types**: "Armenian" or "British" not "the nationality is"

5. **For Titles/Movies/Teams**:
   - Use exact title as written
   - Include quotes, capitalization exactly as shown
   - "Herzeleid" not "Love Metal" or "Herzeleid album"

## Validation Checklist (MUST complete before outputting)
For EVERY answer, self-verify:
- [ ] Did I extract the FULL name with all parts and special characters?
- [ ] Is the capitalization exactly correct?
- [ ] Are quotes included where needed?
- [ ] Is it singular or plural as expected?
- [ ] Is the formatting (en-dash vs hyphen, etc.) matching?
- [ ] Am I not including words like "the", "is", "an", "from the text"?
- [ ] Am I NOT adding extra info (countries, cities, etc.) when not expected?
- [ ] Does my answer match exactly what I computed?

## Synthesis Guidelines
- Combine information from both summaries if they mention different aspects
- IF summaries AGREE on the answer, use that
- IF summaries differ, derive the most complete answer but STOP at the exact expected form
- Match the expected format: "Braunschweig, Lower Saxony" not "Braunschweig, Lower Saxony, Germany"

## Final Output Structure
[[ ## reasoning ## ]]
{reasoning}
[[ ## answer ## ]]
{answer}
[[ ## completed ## ]]

[[ ## reasoning ## ]]
{reasoning}
[[ ## answer ## ]]
{answer}
[[ ## completed ## ]]
