from pydantic import BaseModel


class ReturnProcessingResponse(BaseModel):
    TicketID: str
    RefundStatus: str
    TicketType: str
    Observations: str
    Confidence: float = 1.0


class ReturnProcessingData(BaseModel):
    query: str
    order_id: str
    return_reason: str | None = None
    return_preference: str | None = None
    customer_name: str | None = None
    customer_email: str | None = None
    response: ReturnProcessingResponse
    testcase: str


# Sample return processing data
return_processing_data: list[ReturnProcessingData] = [
    ReturnProcessingData(
        query="I received the wrong item in Order #12345, please fix this.",
        order_id="12345",
        return_reason="wrong item",
        response=ReturnProcessingResponse(
            TicketID="12345",
            RefundStatus="Processing",
            TicketType="Return",
            Observations="Customer received wrong item",
            Confidence=1.0,
        ),
        testcase="Wrong_Item_in_Order",
    ),
    ReturnProcessingData(
        query="I received a damaged item with Order #98765, need a replacement.",
        order_id="98765",
        return_reason="damaged item",
        return_preference="replacement",
        response=ReturnProcessingResponse(
            TicketID="98765",
            RefundStatus="Processing",
            TicketType="Replacement",
            Observations="Customer received damaged item",
            Confidence=1.0,
        ),
        testcase="Damaged_Item_in_Order",
    ),
    ReturnProcessingData(
        query="My order #54321 never arrived, I'd like a refund please.",
        order_id="54321",
        return_reason="order not received",
        return_preference="refund",
        response=ReturnProcessingResponse(
            TicketID="54321",
            RefundStatus="Processing",
            TicketType="Refund",
            Observations="Customer never received order",
            Confidence=1.0,
        ),
        testcase="Order_Not_Received",
    ),
    ReturnProcessingData(
        query="I want to return item from order #78901, it doesn't work properly.",
        order_id="78901",
        return_reason="item malfunction",
        response=ReturnProcessingResponse(
            TicketID="78901",
            RefundStatus="Processing",
            TicketType="Return",
            Observations="Customer reported item malfunction",
            Confidence=1.0,
        ),
        testcase="Item_Malfunction",
    ),
    ReturnProcessingData(
        query="Need to exchange size M for size L on order #24680.",
        order_id="24680",
        return_reason="wrong size",
        return_preference="exchange",
        response=ReturnProcessingResponse(
            TicketID="24680",
            RefundStatus="Processing",
            TicketType="Exchange",
            Observations="Customer needs different size",
            Confidence=1.0,
        ),
        testcase="Wrong_Size_Exchange",
    ),
    ReturnProcessingData(
        query="Order #13579 has missing parts, please help.",
        order_id="13579",
        return_reason="missing parts",
        response=ReturnProcessingResponse(
            TicketID="13579",
            RefundStatus="Processing",
            TicketType="Parts Replacement",
            Observations="Customer reported missing parts",
            Confidence=1.0,
        ),
        testcase="Missing_Parts",
    ),
    ReturnProcessingData(
        query="I didn't like the color of item in order #86420, want to return.",
        order_id="86420",
        return_reason="color preference",
        response=ReturnProcessingResponse(
            TicketID="86420",
            RefundStatus="Processing",
            TicketType="Return",
            Observations="Customer dissatisfied with color",
            Confidence=1.0,
        ),
        testcase="Color_Preference_Return",
    ),
    ReturnProcessingData(
        query="Wrong address on order #97531, need item reshipped.",
        order_id="97531",
        return_reason="wrong address",
        return_preference="reship",
        response=ReturnProcessingResponse(
            TicketID="97531",
            RefundStatus="Processing",
            TicketType="Reship",
            Observations="Order delivered to wrong address",
            Confidence=1.0,
        ),
        testcase="Wrong_Address_Reship",
    ),
    ReturnProcessingData(
        query="I need to return order #35791 to international address, what's the process?",
        order_id="35791",
        return_reason="international return",
        response=ReturnProcessingResponse(
            TicketID="35791",
            RefundStatus="Processing",
            TicketType="International Return",
            Observations="Customer requires international return shipping label",
            Confidence=1.0,
        ),
        testcase="International_Return",
    ),
    ReturnProcessingData(
        query="I bought this 40 days ago with order #46802, is it too late to return?",
        order_id="46802",
        return_reason="late return request",
        response=ReturnProcessingResponse(
            TicketID="46802",
            RefundStatus="Review",
            TicketType="Exception Return",
            Observations="Customer requesting return beyond standard return window",
            Confidence=0.85,
        ),
        testcase="Late_Return_Request",
    ),
    ReturnProcessingData(
        query="I want to return all 5 items from order #27513, they don't match what I expected.",
        order_id="27513",
        return_reason="bulk return",
        return_preference="full refund",
        response=ReturnProcessingResponse(
            TicketID="27513",
            RefundStatus="Processing",
            TicketType="Bulk Return",
            Observations="Customer returning all 5 items from order",
            Confidence=1.0,
        ),
        testcase="Bulk_Order_Return",
    ),
    ReturnProcessingData(
        query="For order #19284, I'd like to keep the headphones but return the case.",
        order_id="19284",
        return_reason="partial return",
        response=ReturnProcessingResponse(
            TicketID="19284",
            RefundStatus="Partial",
            TicketType="Partial Return",
            Observations="Customer keeping headphones but returning case",
            Confidence=0.95,
        ),
        testcase="Partial_Order_Return",
    ),
]


def get_return_processing(
    in_OrderID: str,  # noqa: N803
    in_ReturnReason: str | None = None,  # noqa: N803
    in_ReturnPreference: str | None = None,  # noqa: N803
    in_CustomerName: str | None = None,  # noqa: N803
    in_CustomerEmail: str | None = None,  # noqa: N803
) -> ReturnProcessingResponse:
    """
    Get return processing response based on order ID and return details.

    Args:
        in_OrderID (str): Order ID from customer query
        in_ReturnReason (Optional[str]): Reason for return from customer query
        in_ReturnPreference (Optional[str]): Customer's preference for handling the return
        in_CustomerName (Optional[str]): Customer's name if provided
        in_CustomerEmail (Optional[str]): Customer's email if provided

    Returns:
        ReturnProcessingResponse: Structured return processing response
    """
    # Look for matching order ID in our dataset
    for item in return_processing_data:
        if item.order_id == in_OrderID:
            # Found a match
            return item.response

    # Default response if no match found
    return ReturnProcessingResponse(
        TicketID="",
        RefundStatus="",
        TicketType="",
        Observations="No matching order found",
        Confidence=0.0,
    )
