from pydantic import BaseModel


class DisputeItemsFindings(BaseModel):
    Reasoning: str
    MismatchedParts: str
    ValidReplacement: bool


class DisputeInvestigationResponse(BaseModel):
    SupplierEmail: str
    InvestigationResultString: str
    MismatchedItemsFindingsString: str
    MismatchedItemsFindings: DisputeItemsFindings
    Confidence: float = 1.0


class DisputeInvestigationData(BaseModel):
    query: str
    mismatched_items: str
    company: str | None = None
    supplier: str | None = None
    supplier_website: str | None = None
    response: DisputeInvestigationResponse
    testcase: str


# Sample dispute investigation data
dispute_investigation_data: list[DisputeInvestigationData] = [
    DisputeInvestigationData(
        query="Dispute the following: can Phone X be replaced by Phone Y?",
        mismatched_items="Phone X, Phone Y",
        supplier="PhoneSupplier",
        response=DisputeInvestigationResponse(
            SupplierEmail="a@example.com",
            InvestigationResultString="Phone X and Phone Y are mismatched",
            MismatchedItemsFindingsString="Phone X and Phone Y have different specifications",
            MismatchedItemsFindings=DisputeItemsFindings(
                Reasoning="Phone X and Phone Y have different screen sizes",
                MismatchedParts="screen size",
                ValidReplacement=False,
            ),
            Confidence=1.0,
        ),
        testcase="Phone_Replacement_Dispute",
    ),
    DisputeInvestigationData(
        query="I ordered a Dell XPS 15 but received a Dell Inspiron. Please investigate this dispute.",
        mismatched_items="Dell XPS 15, Dell Inspiron",
        company="TechStore",
        supplier="Dell",
        response=DisputeInvestigationResponse(
            SupplierEmail="dell.support@example.com",
            InvestigationResultString="Dell XPS 15 and Dell Inspiron are different product lines",
            MismatchedItemsFindingsString="Dell XPS 15 is a premium product while Dell Inspiron is a mid-range product",
            MismatchedItemsFindings=DisputeItemsFindings(
                Reasoning="Dell XPS 15 has higher specifications and build quality than Dell Inspiron",
                MismatchedParts="processor, build material, display quality",
                ValidReplacement=False,
            ),
            Confidence=1.0,
        ),
        testcase="Dell_Laptop_Mismatch",
    ),
    DisputeInvestigationData(
        query="I ordered an iPhone 13 Pro but received an iPhone 13. I want to dispute this.",
        mismatched_items="iPhone 13 Pro, iPhone 13",
        supplier="Apple Reseller",
        response=DisputeInvestigationResponse(
            SupplierEmail="apple.reseller@example.com",
            InvestigationResultString="iPhone 13 Pro and iPhone 13 are different models",
            MismatchedItemsFindingsString="iPhone 13 Pro has additional features not present in iPhone 13",
            MismatchedItemsFindings=DisputeItemsFindings(
                Reasoning="iPhone 13 Pro has ProMotion display, telephoto camera, and more RAM",
                MismatchedParts="display refresh rate, camera system, RAM",
                ValidReplacement=False,
            ),
            Confidence=1.0,
        ),
        testcase="iPhone_Model_Dispute",
    ),
    DisputeInvestigationData(
        query="Please investigate this dispute: I ordered Nike Air Max but received Nike Revolution shoes.",
        mismatched_items="Nike Air Max, Nike Revolution",
        supplier="SportsShop",
        response=DisputeInvestigationResponse(
            SupplierEmail="sportsshop@example.com",
            InvestigationResultString="Nike Air Max and Nike Revolution are different shoe models",
            MismatchedItemsFindingsString="Nike Air Max is a premium model with different technology than Nike Revolution",
            MismatchedItemsFindings=DisputeItemsFindings(
                Reasoning="Nike Air Max uses Air cushioning technology while Nike Revolution uses different foam",
                MismatchedParts="cushioning technology, price point, design",
                ValidReplacement=False,
            ),
            Confidence=1.0,
        ),
        testcase="Nike_Shoes_Dispute",
    ),
    DisputeInvestigationData(
        query="I received a Samsung S22 instead of Samsung S22 Ultra. Need to dispute this.",
        mismatched_items="Samsung S22, Samsung S22 Ultra",
        supplier="MobileWorld",
        response=DisputeInvestigationResponse(
            SupplierEmail="mobileworld@example.com",
            InvestigationResultString="Samsung S22 and Samsung S22 Ultra are different models",
            MismatchedItemsFindingsString="Samsung S22 Ultra has additional features not present in Samsung S22",
            MismatchedItemsFindings=DisputeItemsFindings(
                Reasoning="Samsung S22 Ultra has S Pen support, larger screen, better cameras, and larger battery",
                MismatchedParts="display size, camera system, S Pen compatibility, battery capacity",
                ValidReplacement=False,
            ),
            Confidence=1.0,
        ),
        testcase="Samsung_Phone_Dispute",
    ),
]


def get_dispute_investigation(
    in_MismatchedItems: str,  # noqa: N803
    in_Company: str | None = None,  # noqa: N803
    in_Supplier: str | None = None,  # noqa: N803
    in_SupplierWebsite: str | None = None,  # noqa: N803
) -> DisputeInvestigationResponse:
    """
    Get dispute investigation response based on mismatched items and company/supplier information.

    Args:
        in_MismatchedItems (str): The mismatched items from customer query
        in_Company (Optional[str]): The company that received the mismatched items
        in_Supplier (Optional[str]): The supplier who delivered the mismatched items
        in_SupplierWebsite (Optional[str]): The supplier's website that contains their item catalog

    Returns:
        DisputeInvestigationResponse: Structured dispute investigation response
    """
    # Look for matching items in our dataset
    for item in dispute_investigation_data:
        # Check if all parts of the mismatched items are in the query
        mismatched_parts = item.mismatched_items.split(", ")
        if all(part.lower() in in_MismatchedItems.lower() for part in mismatched_parts):
            # Found a match
            return item.response

    # Default response if no match found - empty response
    return DisputeInvestigationResponse(
        SupplierEmail="",
        InvestigationResultString="",
        MismatchedItemsFindingsString="",
        MismatchedItemsFindings=DisputeItemsFindings(
            Reasoning="", MismatchedParts="", ValidReplacement=False
        ),
        Confidence=0.0,
    )
