from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class LoanAgent_ApplyUiBankLoanInput(BaseModel):  # noqa: N801
    RequestorEmailAddress: str = Field(description="The email of the requester")
    LoanAmount: int = Field(description="The loan amount requested (integer)")
    LoanTerm: int = Field(description="The loan term (integer)")
    Income: int = Field(description="The requester's income (integer)")
    Age: int = Field(description="The requester's age (integer)")


class LoanAgent_ModifyLoanContractInput(BaseModel):  # noqa: N801
    SharepointFolderURL: str = Field(description="SharePoint URL")
    InterestRate: float = Field(description="The interest rate of the loan")
    LoanAmount: int = Field(description="The loan amount")
    BorrowerName: str = Field(description="The borrower name")
    BorrowerAddress: str | None = Field(description="The borrower address")


class LoanAgent_SendSharepointDocumentViaDocuSignInput(BaseModel):  # noqa: N801
    SharepointFileURL: str = Field(
        description="The SharePoint URL of the file created by the LoanAgent_ModifyLoanContract tool"
    )
    RecipientLegalName: str = Field(description="The recipient legal name")
    RecipientEmail: str = Field(description="The recipient email")


class GoogleInput(BaseModel):
    SearchText: str = Field(description="Search text")


@tool
async def LoanAgent_ApplyUiBankLoan(  # noqa: N802
    RequestorEmailAddress: Annotated[str, "The email of the requester"],  # noqa: N803
    LoanAmount: Annotated[int, "The loan amount requested"],  # noqa: N803
    LoanTerm: Annotated[int, "The loan term"],  # noqa: N803
    Income: Annotated[int, "The requester's income"],  # noqa: N803
    Age: Annotated[int, "The requester's age"],  # noqa: N803
) -> str:
    """Apply for a loan at UI Bank.

    Args:
        RequestorEmailAddress: The email of the requester
        LoanAmount: The loan amount requested (integer)
        LoanTerm: The loan term (integer)
        Income: The requester's income (integer)
        Age: The requester's age (integer)

    Returns:
        string: Loan application details
    """
    if LoanTerm == 5 or LoanTerm == 3 or LoanTerm == 1:
        if Age >= 18:
            return "Loan application was approved with a rate of 4% APR."
        else:
            return "The age of the applicant is below the minimum age for a loan."
    else:
        return "Loan application term is not supported."


@tool
async def LoanAgent_ModifyLoanContract(  # noqa: N802
    SharepointFolderURL: Annotated[str, "SharePoint URL"],  # noqa: N803
    InterestRate: Annotated[float, "The interest rate of the loan"],  # noqa: N803
    LoanAmount: Annotated[int, "The loan amount"],  # noqa: N803
    BorrowerName: Annotated[str, "The borrower name"],  # noqa: N803
    BorrowerAddress: Annotated[str, "The borrower address"],  # noqa: N803
) -> str:
    """Modify a loan contract.

    Args:
        SharepointFolderURL: The SharePoint URL of the file
        InterestRate: The interest rate of the loan
        LoanAmount: The loan amount
        BorrowerName: The borrower name
        BorrowerAddress: The borrower address

    Returns:
        str: Sharepoint file URL
    """
    # Mock tool: returns a fixed file name in the given folder.
    return SharepointFolderURL + "/New_doc.pdf"


@tool
async def LoanAgent_SendSharepointDocumentViaDocuSign(  # noqa: N802
    SharepointFileURL: Annotated[  # noqa: N803
        str,
        "The SharePoint URL of the file",
    ],
    RecipientLegalName: Annotated[str, "The recipient legal name"],  # noqa: N803
    RecipientEmail: Annotated[str, "The recipient email"],  # noqa: N803
) -> str:
    """Send a sharepoint document via DocuSign.

    Args:
        SharepointFileURL: The SharePoint URL of the file
        RecipientLegalName: The recipient legal name
        RecipientEmail: The recipient email

    Returns:
        str: True if document is sent successfully, False otherwise
    """
    # Mock tool: returns a fixed value.
    return "Document New_doc.pdf was sent to: random@example.com"


@tool
async def Google(SearchText: Annotated[str, "Search text"]) -> str:  # noqa: N802, N803
    """Search the web for a given text.

    Args:
        SearchText: The text to search for

    Returns:
        str: Search results
    """
    return "The National Bank kept its benchmark interest rate unchanged at 6.5% during its meeting, following two consecutive rate cuts."
