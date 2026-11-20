from pydantic import BaseModel


class ProductItem(BaseModel):
    Product: str
    Price: str


class ProductRecommendationsResponse(BaseModel):
    Recommendations: dict
    Confidence: float = 1.0
    AdditionalInfo: str = ""


class ProductRecommendationData(BaseModel):
    product_context: str
    customer_preferences: str
    response: ProductRecommendationsResponse
    testcase: str


# Sample product recommendations data
product_recommendations_data: list[ProductRecommendationData] = [
    ProductRecommendationData(
        product_context="compact camera",
        customer_preferences="budget-friendly, portable",
        response=ProductRecommendationsResponse(
            Recommendations={
                "TopMatch": {
                    "Product": "Canon PowerShot SX620 HS",
                    "Price": "$199.00",
                },
                "Alternatives": [
                    {
                        "Product": "Sony Cyber-shot DSC-W800",
                        "Price": "$99.00",
                    },
                    {
                        "Product": "Nikon Coolpix B500",
                        "Price": "$149.00",
                    },
                ],
            },
            AdditionalInfo="These compact cameras offer excellent portability and image quality at different price points.",
        ),
        testcase="compact_camera_recommendation",
    ),
    ProductRecommendationData(
        product_context="smartphone",
        customer_preferences="high performance, good camera",
        response=ProductRecommendationsResponse(
            Recommendations={
                "TopMatch": {
                    "Product": "Samsung Galaxy S21",
                    "Price": "$799.00",
                },
                "Alternatives": [
                    {
                        "Product": "iPhone 12",
                        "Price": "$899.00",
                    },
                    {
                        "Product": "Google Pixel 5",
                        "Price": "$699.00",
                    },
                ],
            },
            AdditionalInfo="These smartphones all feature excellent cameras and high-performance processors.",
        ),
        testcase="smartphone_recommendation",
    ),
    ProductRecommendationData(
        product_context="laptop",
        customer_preferences="gaming, high performance",
        response=ProductRecommendationsResponse(
            Recommendations={
                "TopMatch": {
                    "Product": "ASUS ROG Zephyrus G14",
                    "Price": "$1,299.00",
                },
                "Alternatives": [
                    {
                        "Product": "Razer Blade 15",
                        "Price": "$1,699.00",
                    },
                    {
                        "Product": "MSI GS66 Stealth",
                        "Price": "$1,499.00",
                    },
                ],
            },
            AdditionalInfo="These gaming laptops offer excellent performance with dedicated GPUs and high refresh rate displays.",
        ),
        testcase="gaming_laptop_recommendation",
    ),
    ProductRecommendationData(
        product_context="headphones",
        customer_preferences="wireless, noise cancellation",
        response=ProductRecommendationsResponse(
            Recommendations={
                "TopMatch": {
                    "Product": "Sony WH-1000XM4",
                    "Price": "$349.00",
                },
                "Alternatives": [
                    {
                        "Product": "Bose QuietComfort 45",
                        "Price": "$329.00",
                    },
                    {
                        "Product": "Apple AirPods Pro",
                        "Price": "$249.00",
                    },
                ],
            },
            AdditionalInfo="These wireless headphones offer industry-leading noise cancellation and excellent sound quality.",
        ),
        testcase="wireless_headphones_recommendation",
    ),
    ProductRecommendationData(
        product_context="smartwatch",
        customer_preferences="fitness tracking, long battery life",
        response=ProductRecommendationsResponse(
            Recommendations={
                "TopMatch": {
                    "Product": "Garmin Forerunner 245",
                    "Price": "$299.00",
                },
                "Alternatives": [
                    {
                        "Product": "Fitbit Versa 3",
                        "Price": "$229.00",
                    },
                    {
                        "Product": "Apple Watch SE",
                        "Price": "$279.00",
                    },
                ],
            },
            AdditionalInfo="These smartwatches excel at fitness tracking with various sensors and offer good battery life.",
        ),
        testcase="fitness_smartwatch_recommendation",
    ),
    ProductRecommendationData(
        product_context="tablet",
        customer_preferences="productivity, stylus support",
        response=ProductRecommendationsResponse(
            Recommendations={
                "TopMatch": {
                    "Product": "iPad Pro 11-inch",
                    "Price": "$799.00",
                },
                "Alternatives": [
                    {
                        "Product": "Samsung Galaxy Tab S7",
                        "Price": "$649.00",
                    },
                    {
                        "Product": "Microsoft Surface Pro 7",
                        "Price": "$749.00",
                    },
                ],
            },
            AdditionalInfo="These tablets support stylus input and are excellent for productivity with optional keyboard accessories.",
        ),
        testcase="productivity_tablet_recommendation",
    ),
    ProductRecommendationData(
        product_context="tv",
        customer_preferences="4K, smart features",
        response=ProductRecommendationsResponse(
            Recommendations={
                "TopMatch": {
                    "Product": "LG OLED C1 Series 65-inch",
                    "Price": "$1,799.00",
                },
                "Alternatives": [
                    {
                        "Product": "Samsung QN90A Neo QLED 55-inch",
                        "Price": "$1,299.00",
                    },
                    {
                        "Product": "Sony X90J 65-inch",
                        "Price": "$1,199.00",
                    },
                ],
            },
            AdditionalInfo="These 4K smart TVs offer excellent picture quality and comprehensive smart features.",
        ),
        testcase="smart_tv_recommendation",
    ),
    ProductRecommendationData(
        product_context="printer",
        customer_preferences="wireless, all-in-one",
        response=ProductRecommendationsResponse(
            Recommendations={
                "TopMatch": {
                    "Product": "HP OfficeJet Pro 9015e",
                    "Price": "$229.00",
                },
                "Alternatives": [
                    {
                        "Product": "Canon PIXMA TR8620",
                        "Price": "$179.00",
                    },
                    {
                        "Product": "Epson WorkForce Pro WF-4830",
                        "Price": "$199.00",
                    },
                ],
            },
            AdditionalInfo="These all-in-one wireless printers offer printing, scanning, and copying capabilities with mobile connectivity.",
        ),
        testcase="wireless_printer_recommendation",
    ),
    ProductRecommendationData(
        product_context="bluetooth speaker",
        customer_preferences="portable, waterproof",
        response=ProductRecommendationsResponse(
            Recommendations={
                "TopMatch": {
                    "Product": "JBL Flip 5",
                    "Price": "$119.00",
                },
                "Alternatives": [
                    {
                        "Product": "Ultimate Ears BOOM 3",
                        "Price": "$149.00",
                    },
                    {
                        "Product": "Bose SoundLink Flex",
                        "Price": "$149.00",
                    },
                ],
            },
            AdditionalInfo="These portable Bluetooth speakers are waterproof and offer excellent sound quality for their size.",
        ),
        testcase="portable_speaker_recommendation",
    ),
    ProductRecommendationData(
        product_context="monitor",
        customer_preferences="gaming, high refresh rate",
        response=ProductRecommendationsResponse(
            Recommendations={
                "TopMatch": {
                    "Product": "LG 27GL850-B 27-inch",
                    "Price": "$449.00",
                },
                "Alternatives": [
                    {
                        "Product": "ASUS TUF Gaming VG27AQ 27-inch",
                        "Price": "$379.00",
                    },
                    {
                        "Product": "Samsung Odyssey G7 32-inch",
                        "Price": "$699.00",
                    },
                ],
            },
            AdditionalInfo="These gaming monitors feature high refresh rates and low response times for competitive gaming.",
        ),
        testcase="gaming_monitor_recommendation",
    ),
    ProductRecommendationData(
        product_context="vacuum cleaner",
        customer_preferences="robot, smart mapping",
        response=ProductRecommendationsResponse(
            Recommendations={
                "TopMatch": {
                    "Product": "iRobot Roomba j7+",
                    "Price": "$799.00",
                },
                "Alternatives": [
                    {
                        "Product": "Roborock S7",
                        "Price": "$649.00",
                    },
                    {
                        "Product": "Shark IQ Robot XL",
                        "Price": "$599.00",
                    },
                ],
            },
            AdditionalInfo="These robot vacuums feature smart mapping technology and can navigate efficiently around your home.",
        ),
        testcase="robot_vacuum_recommendation",
    ),
    ProductRecommendationData(
        product_context="coffee maker",
        customer_preferences="programmable, large capacity",
        response=ProductRecommendationsResponse(
            Recommendations={
                "TopMatch": {
                    "Product": "Cuisinart DCC-3200P1",
                    "Price": "$99.00",
                },
                "Alternatives": [
                    {
                        "Product": "Ninja CE251 Programmable Brewer",
                        "Price": "$79.00",
                    },
                    {
                        "Product": "Hamilton Beach 12-Cup",
                        "Price": "$59.00",
                    },
                ],
            },
            AdditionalInfo="These programmable coffee makers offer large capacity brewing with customizable settings.",
        ),
        testcase="coffee_maker_recommendation",
    ),
    # Add more product recommendation data as needed
]


def get_product_recommendations(
    in_ProductContext: str,  # noqa: N803
    in_CustomerPreferences: str,  # noqa: N803
) -> ProductRecommendationsResponse:
    """
    Get product recommendations based on product context and customer preferences.

    Args:
        in_ProductContext (str): Product context from customer query
        in_CustomerPreferences (str): Customer preferences for the product

    Returns:
        ProductRecommendationsResponse: Structured recommendations object
    """
    # Look for matching recommendations in our dataset
    for item in product_recommendations_data:
        if item.product_context in in_ProductContext.lower():
            # Found a match
            return item.response

    # Default empty response if no match found
    return ProductRecommendationsResponse(
        Recommendations={},
        Confidence=0.0,
        AdditionalInfo="No specific recommendations found for the requested product.",
    )
