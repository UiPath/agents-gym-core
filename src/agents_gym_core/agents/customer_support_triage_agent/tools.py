from typing import Annotated

from langchain_core.tools import tool

from agents_gym_core.agents.customer_support_triage_agent.mock_data.dispute_investigation_agent import (
    DisputeInvestigationResponse,
    get_dispute_investigation,
)
from agents_gym_core.agents.customer_support_triage_agent.mock_data.product_questions_agent import (
    ProductQuestionsResponse,
    get_product_response,
)
from agents_gym_core.agents.customer_support_triage_agent.mock_data.product_recommendation_agent import (
    ProductRecommendationsResponse,
    get_product_recommendations,
)
from agents_gym_core.agents.customer_support_triage_agent.mock_data.return_processing_agent import (
    ReturnProcessingResponse,
    get_return_processing,
)


@tool
async def Product_Recommendation_Agent(  # noqa: N802
    in_ProductContext: Annotated[str, "Details of the product extracted from <Request>."],  # noqa: N803
    in_CustomerPreferences: Annotated[  # noqa: N803
        str,
        "Specific customer preferences that are not directly mapped to a product from <Request>.",
    ],
) -> ProductRecommendationsResponse:
    """
    Assists customers in finding the right products based on their needs and preferences.
    This agent specializes in offering personalized suggestions to enhance customer satisfaction.
    """
    return get_product_recommendations(in_ProductContext, in_CustomerPreferences)


@tool
async def Return_Processing_Agent(  # noqa: N802
    in_OrderID: Annotated[str, "The order id of the product if it was specified in <Request>"],  # noqa: N803
    in_ReturnReason: Annotated[str | None, "The returning reason extracted from <Request>"] = None,  # noqa: N803
    in_ReturnPreference: Annotated[  # noqa: N803
        str | None, "The returning preferences extracted from <Request>"
    ] = None,
    out_Observations: Annotated[  # noqa: N803
        str | None,
        "Observations to be returned to the customer if they were specified in <Request>",
    ] = None,
    out_TicketType: Annotated[  # noqa: N803
        str | None, "Ticket type to be returned to the customer if they were specified in <Request>"
    ] = None,
    in_CustomerName: Annotated[  # noqa: N803
        str | None, "Customer name if they were specified in <Request>"
    ] = None,
    in_CustomerEmail: Annotated[  # noqa: N803
        str | None, "Customer email if they were specified in <Request>"
    ] = None,
) -> ReturnProcessingResponse:
    """
    Manages customer requests related to returning products, including initiating returns and tracking return shipments.
    """
    return get_return_processing(
        in_OrderID, in_ReturnReason, in_ReturnPreference, in_CustomerName, in_CustomerEmail
    )


@tool
async def Product_Questions_Agent(  # noqa: N802
    in_CustomerQuery: Annotated[str, "<Request>"],  # noqa: N803
    in_ProductName: Annotated[str, "Product name extracted from <Request>"],  # noqa: N803
    in_ProductManufacturer: Annotated[  # noqa: N803
        str | None, "Product Manufacturer extracted from <Request>"
    ] = None,
) -> ProductQuestionsResponse | str:
    """
    Handles all customer questions about product specifications, availability, and usage instructions.
    This agent focuses on providing accurate and detailed information about the product.
    """
    try:
        return get_product_response(in_CustomerQuery, in_ProductName, in_ProductManufacturer)
    except Exception as e:
        return str(e)


@tool
async def Dispute_Investigation_Agent(  # noqa: N802
    in_MismatchedItems: Annotated[str, "The mismatched items from <Request>"],  # noqa: N803
    in_Company: Annotated[  # noqa: N803
        str | None, "The company extracted from <Request> who received the mismatched items"
    ] = None,
    in_Supplier: Annotated[  # noqa: N803
        str | None, "The supplier extracted from <Request> who delivered the mismatched items"
    ] = None,
    in_SupplierWebsite: Annotated[  # noqa: N803
        str | None,
        "The supplier's website extracted from <Request> that contains their item catalog",
    ] = None,
) -> DisputeInvestigationResponse:
    """
    Handles disputes, aiming to resolve issues amicably and improve customer satisfaction.
    This agent focuses on gathering the case information and presenting it to human-in-the-loop.
    """
    return get_dispute_investigation(
        in_MismatchedItems, in_Company, in_Supplier, in_SupplierWebsite
    )


@tool
async def Escalation_Agent(  # noqa: N802
    in_Query: Annotated[str, "The query or observation that needs to be escalated to a human."],  # noqa: N803
) -> str:
    """
    Raises queries to a human that are not appropriate for other agents.
    Such queries may include those that are too vague or too complex for the other agents.
    """
    return f"Escalated to a human. Query: {in_Query}"
