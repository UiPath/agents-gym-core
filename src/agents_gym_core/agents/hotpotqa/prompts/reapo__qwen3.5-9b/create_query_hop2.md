# Task: Generate a precise query for Wikipedia search

Given the fields `question` and `summary_1`, generate a focused query to find EXACT missing information needed to answer the question.

## Purpose
Create queries that target SPECIFIC missing facts without being too broad or vague.

## Query Generation Rules
1. **Build from what's known**: 
   - List what summary_1 tells us about the answer
   - Identify exactly what's MISSING
   - Create a query that finds the missing piece specifically

2. **Be precise, not general**:
   - BAD: "Who is related to X?" 
   - GOOD: "X relative _father_ or descendant year"
   - BAD: "Book about Titanic"
   - GOOD: "Titanic book author 1950s Canadian"

3. **Use full terms from context**:
   - If summary mentions "Ralph Waldo Emerson", use that FULL name in query
   - If summary mentions "The United States Naval Academy", use that EXACT phrase
   - Include ALL identifying details (years, locations, affiliations)

4. **Combine knowledge into a hop**:
   - From clue A + clue B → generate query for finding C
   - From partial answer + missing piece → generate query for completing

5. **Query length**: 3-8 words - specific enough to retrieve exact answer

## Failure Mode Prevention
- Don't create queries that are too broad ("who won world cup")
- Don't repeat what's already answered
- Don't forget key identifiers present in summary_1

## Output Requirements
- reasoning: What specific fact is still needed? Why the current query will find it?
- query: Precise search terms that target the missing information
