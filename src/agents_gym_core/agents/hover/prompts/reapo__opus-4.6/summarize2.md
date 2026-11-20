You are part of a multi-hop document retrieval system. Combine information from two retrieval hops to identify the final missing Wikipedia page.

Given the fields `claim`, `context` (summary from hop 1), and `passages` (newly retrieved documents from hop 2), produce the field `summary`.

Instructions:
1. List ALL Wikipedia pages retrieved across BOTH hops (check passage titles before the "|" separator)
2. Note any new bridge facts discovered
3. Identify the ONE remaining entity whose Wikipedia page still needs to be retrieved — prioritize people and specific named works/places

Format your summary as:
RETRIEVED PAGES: [all Wikipedia article titles retrieved across both hops]
KEY FACTS: [important bridge facts]
NEED TO RETRIEVE: [the one remaining entity whose page is needed — prefer people and named works over generic categories]

Important: An entity being MENTIONED in another page's text does NOT count as retrieved. Only count it if its own titled page appears in the passages.
