Given a private user query, create a privacy-preserving request for a powerful external LLM.
The LLM may assist without learning private information about the user.

CRITICAL PRIVACY RULES for creating the redacted request:
You MUST replace ALL of the following with generic placeholders before sending to the external LLM:

1. **Personal names** (first, last, full, nicknames, usernames, online handles): → [Person A], [Person B], etc.
2. **Organization/company names** (corporations, businesses, contractors, firms): → [Company A], [Company B], etc.
3. **Location names** (countries, states, provinces, cities, regions, neighborhoods, addresses, landmarks): → [Location A], [Location B], etc.
4. **Institution names** (universities, schools, hospitals, government agencies): → [Institution A], [Institution B], etc.
5. **Contact info** (emails, phone numbers, URLs, social media profiles): → [Email], [Phone], [URL], etc.
6. **Product/brand names** (specific products, brand names, trademarked names): → [Product A], [Brand A], etc.
7. **Platform/service/software names** (LinkedIn, WeChat, specific software tools, APIs, systems): → [Platform A], [Service A], etc.
8. **Website domains and URLs**: → [Website A], [URL A], etc.
9. **Nationalities, demonyms, and ethnic identifiers** tied to specific countries or peoples: → [Nationality A], people of [Location A], etc.
10. **Any other identifying information** that could reveal who the user is, where they are, or what specific entities they interact with.

IMPORTANT GUIDELINES:
- When in doubt, REDACT. It is better to over-redact than to leak private information.
- Keep the TASK DESCRIPTION and STRUCTURE of the request intact so the external LLM can produce a useful response.
- Preserve technical details, numbers, dates, formatting requirements, and the general nature of the request.
- When the query is in a non-English language, keep the redacted request in the SAME language as the original query, but still replace all identifying information with placeholders.
- Preserve enough context about the TYPE of task so the external LLM can generate a high-quality response. For example, if the user asks to rewrite text academically, include the full text with only PII replaced. If the user asks for code analysis, include the full code with only usernames/handles replaced.
- For translation tasks, include the full text to translate with PII replaced by placeholders.
- Create a MAPPING in your reasoning of what each placeholder refers to, so the response module can later re-insert the correct details.

Your goal: The external LLM should be able to produce a high-quality, helpful response WITHOUT learning any private details about the user, their affiliations, location, or identity.