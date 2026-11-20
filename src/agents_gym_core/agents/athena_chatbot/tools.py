from typing import Annotated

from langchain_core.tools import tool


@tool
async def tool_send_mentorship_confirmation(
    in_userEmail: Annotated[str, "The user email of the mentor."],  # noqa: N803
    in_textOfMessage: Annotated[  # noqa: N803
        str,
        "A text message containing the details about the mentee to whom this mentor will be matched.",
    ],
    in_isUat: Annotated[  # noqa: N803
        bool,
        "Whether the message is to be sent in UAT or not. If it is, then True, otherwise False.",
    ] = False,
) -> str:
    """Sends a slack confirmation message to mentors, asking if they want to approve or reject mentorship request."""
    return (
        f"Sent mentorship confirmation message to {in_userEmail} with message: {in_textOfMessage}."
    )
