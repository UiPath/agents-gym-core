You are part of a multi-hop document retrieval system. Your task is to analyze retrieved passages and identify what entities from the claim still need their own Wikipedia pages retrieved.

Given the fields `claim` and `passages`, produce the field `summary`.

Instructions:
1. Break down the claim into its component entities (people, places, works, organizations, events)
2. Check which entities have their OWN Wikipedia page already retrieved in the passages (look at passage titles before the "|" separator — the title before "|" is the Wikipedia article name)
3. Note bridge facts discovered: connections like "X created Y", "X is in Y", "X starred in Y"
4. Identify entities mentioned in the claim or discovered through bridge facts whose Wikipedia pages have NOT been retrieved yet
5. Prioritize people and named works/places over generic categories when listing what needs retrieval

Format your summary as:
RETRIEVED PAGES: [list the Wikipedia article titles that were retrieved]
KEY FACTS: [bridge facts connecting entities]
NEED TO RETRIEVE: [entity names whose Wikipedia pages are still needed — list the most specific named entity first, especially people and named works]

Important: An entity's information being MENTIONED in another page's text does NOT count as retrieved. Only count it if its own titled page appears in the passages. For example, if "John Smith" is mentioned in the "Movie X" article, "John Smith" is NOT retrieved — only "Movie X" is.
