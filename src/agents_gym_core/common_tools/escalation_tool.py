import re
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing_extensions import override

META_DESCRIPTION_ESCALATION_TOOL: str = """This is an escalation. This tool must be called separately from other tool calls (i.e. not in parallel with other tool calls).
This is because you must wait for the response before deciding what tool to call next.
"""


class EscalationToolInput(BaseModel):
    query: str = Field(..., title="Query string")


class ReturnRule(BaseModel):
    """A rule for conditional return messages.

    Attributes:
        match: Regex pattern to match against the query.
        return_message: Message to return when matched. Supports {query} placeholder.
    """

    match: str
    return_message: str


class EscalationTool(BaseTool):
    """Escalation tool that simulates human-in-the-loop intervention.

    Supports three response modes (checked in order):
      1. return_rules: List of {match, return_message} rules. The query is checked
         against each rule's regex pattern in order; the first match wins.
         The return_message supports a {query} placeholder.
      2. return_message: A static (or {query}-templated) fallback message.
      3. Default: A generic "task created" message.
    """

    name: str = "escalation_tool"
    description: str
    assign_to: str
    return_message: str | None = None
    return_rules: list[ReturnRule] | None = None

    def __init__(
        self,
        description: str = "",
        **kwargs: Any,
    ):
        super().__init__(
            description=f"{META_DESCRIPTION_ESCALATION_TOOL}{description}",
            args_schema=EscalationToolInput,
            **kwargs,
        )

    @override
    async def _arun(self, query: str) -> str:
        """Async run method required by BaseTool"""
        return self._run(query)

    @override
    def _run(self, query: str) -> str:
        """Synchronous run method required by BaseTool"""
        if self.return_rules:
            for rule in self.return_rules:
                if re.search(rule.match, query, re.IGNORECASE | re.DOTALL):
                    return rule.return_message.replace("{query}", query)

        if self.return_message:
            return self.return_message.replace("{query}", query)

        return f'The escalation was successful. A task has been created for this query: "{query}", and assigned to {self.assign_to}. Call the end execution tool.'
