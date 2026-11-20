Summarize the conversation above. The original messages will be discarded, so the summary must be self-contained. Keep it as short as possible — the summary MUST be shorter than the messages it replaces.

Use this exact structure:

**Task**: Copy the user's original request/input EXACTLY as given — do NOT paraphrase or shorten it. This is the source of truth for what needs to be done.
**Results**: Accumulated outputs as compact key→value pairs (verbatim keys and values — do not rephrase). Include ALL specific values (including scores and numeric metadata) needed to make decisions or build the final response. No full sentences, just key→value.
**Calls**: Tool calls already made as `tool(arg)→outcome` one-liners, so the same call is not repeated.
**Pending**: First, list any tool call from the agent workflow that is explicitly marked MUST but does not appear in **Calls**. Then, list any item in **Task** without a matching entry in **Results**. Only write "None" if no MUST tools are missing AND every task item has a result.
**Next step**: State the single next action: either the next pending item to work on, or "Call end_execution with the final result" if nothing is pending.