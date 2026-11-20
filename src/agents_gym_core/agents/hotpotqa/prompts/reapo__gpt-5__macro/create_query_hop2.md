Task: Generate the second-hop search query that will retrieve the missing entity/fact to bridge the reasoning chain.

Instructions:
- Use the question and summary_1 to identify the missing piece for hop 2.
- Compose ONE succinct query string (<= 12 tokens) that includes:
  - The key entity from hop 1 (with most specific alias),
  - The target relation or attribute to retrieve,
  - Any disambiguating detail (year, occupation, team, location) that reduces ambiguity.
- Avoid pronouns; use explicit names. No punctuation except spaces. No quotes. No explanations.

Produce the field `query` only.
