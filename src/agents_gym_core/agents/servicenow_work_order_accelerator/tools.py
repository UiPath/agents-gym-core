from langchain_core.tools import tool


@tool
async def Create_New_Incident(  # noqa: N802
    Description: str,  # noqa: N803
    Caller: str,  # noqa: N803
    Urgency: int,  # noqa: N803
    Impact: int,  # noqa: N803
) -> dict[str, str | int]:
    """Create a new incident in ServiceNow"""
    return {
        "IncidentNumber": "9564aa11-8937-4a9c-9c65-00dbc6840a9d",
        "Status": "Succeeded",
        "Urgency": Urgency,
        "Impact": Impact,
        "Caller": Caller,
        "Description": Description,
    }
