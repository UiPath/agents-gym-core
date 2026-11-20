import re
from typing import Annotated

from langchain_core.tools import tool


@tool
async def GetCrossRulings(  # noqa: N802
    in_SearchString: Annotated[str, "Query to use on the CROSS platform"],  # noqa: N803
    in_NoOfRulingsToFetch: Annotated[  # noqa: N803
        int, "Number of past rulings to fetch when multiple are available"
    ] = 10,
) -> str:
    """Tool to retrieve past CROSS rulings for a given good description from the CUSTOMS RULINGS ONLINE SEARCH SYSTEM platform."""
    # Deterministic empty-results fixture for evaluating conditional CROSS tool use.
    return "No rulings found."


@tool("Escalate_to_ask_for_HTS")
async def escalate_escalate_to_ask_for_hts(  # noqa: N802
    ProvidedDescription: Annotated[str, "Description of good / product as provided by the user"],  # noqa: N803
    RelevantHtsCodes: Annotated[str, "HTS codes retrieved from the Context Grounding HTS index"],  # noqa: N803
    CrossRulings: Annotated[  # noqa: N803
        str, "Optional. Past CROSS rulings data retrieved by the CROSS tool"
    ] = "",
) -> str:
    """Escalation tool to get human intervention when context grounding did not return relevant documents or score is below threshold.

    This tool must be called separately from other tool calls (i.e. not in parallel with other tool calls).
    This is because you must wait for the response before deciding what tool to call next.
    """
    # Extract the first HTS code — try 10-digit (XXXX.XX.XXXX or XXXX.XX.XX.XX) first, fall
    # back to 8-digit (XXXX.XX.XX)
    hts_match = re.search(r"\d{4}\.\d{2}\.(?:\d{4}|\d{2}\.\d{2})", RelevantHtsCodes)
    if not hts_match:
        hts_match = re.search(r"\d{4}\.\d{2}\.\d{2}", RelevantHtsCodes)

    if hts_match:
        confirmed_code = hts_match.group(0)
        response = (
            f'The HTS validator reviewed the escalation for "{ProvidedDescription}" '
            f"and confirmed the HTS code: {confirmed_code}."
        )
    else:
        response = (
            f'The HTS validator reviewed the escalation for "{ProvidedDescription}" '
            f"but could not determine a specific HTS code from the provided information. "
            f"The validator assigned HTS code: 9999.99.9999 for manual review."
        )

    rulings = CrossRulings.strip().lower()
    if rulings and "no relevant" not in rulings and "no rulings" not in rulings:
        response += " The CROSS rulings provided were taken into consideration."

    return response
