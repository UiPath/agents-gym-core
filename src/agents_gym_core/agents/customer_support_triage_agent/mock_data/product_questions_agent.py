from collections.abc import Callable

from langchain_core.runnables.config import ensure_config
from pydantic import BaseModel


class ProductQuestionsResponse(BaseModel):
    FinalResponse: str
    QuestionsAnswered: str
    QuestionsToEscalate: str
    Confidence: float


class ProductQuestionsData(BaseModel):
    query: str
    product: str | None = None
    products: list[str] | None = None
    response: ProductQuestionsResponse
    testcase: str


# Convert the raw dictionary data to Pydantic models
product_questions_data: list[ProductQuestionsData] = [
    ProductQuestionsData(
        query="What is the warranty period for Lenovo ThinkPad X1 Carbon?",
        product="Lenovo ThinkPad X1 Carbon",
        response=ProductQuestionsResponse(
            FinalResponse="The warranty period for Lenovo ThinkPad X1 Carbon is 2 years.",
            QuestionsAnswered="",
            QuestionsToEscalate="",
            Confidence=0.87,
        ),
        testcase="laptop_warranty_period",
    ),
    ProductQuestionsData(
        query="What is the price for Phone S20?",
        product="Phone S20",
        response=ProductQuestionsResponse(
            FinalResponse="To provide the price for Phone S20, I need to know the manufacturer of Phone S20.",
            QuestionsAnswered="",
            QuestionsToEscalate="Who is the manufacturer of Phone S20?",
            Confidence=1,
        ),
        testcase="phone_price",
    ),
    ProductQuestionsData(
        query="Can you tell me the difference between iphone 12 and iphone 12 pro? I want to compare your models and see if I have other questions. ?",
        products=["iphone 12", "iphone 12 pro"],
        response=ProductQuestionsResponse(
            FinalResponse="The iPhone 12 Pro offers several advantages over the iPhone 12: 1) Better camera system with telephoto lens and LiDAR scanner, 2) Premium stainless steel frame vs aluminum, 3) Brighter max screen brightness, 4) 6GB RAM vs 4GB RAM, 5) Larger storage options up to 512GB",
            QuestionsAnswered="",
            QuestionsToEscalate="",
            Confidence=1,
        ),
        testcase="iphone_12_vs_12_pro",
    ),
    ProductQuestionsData(
        query="What are the charging options for the Samsung Galaxy S21?",
        product="Samsung Galaxy S21",
        response=ProductQuestionsResponse(
            FinalResponse="The Samsung Galaxy S21 supports 25W fast wired charging, 15W wireless charging, and 4.5W reverse wireless charging.",
            QuestionsAnswered="",
            QuestionsToEscalate="",
            Confidence=0.95,
        ),
        testcase="samsung_galaxy_s21_charging_options",
    ),
    ProductQuestionsData(
        query="Is the Dell XPS 13 compatible with Windows 11?",
        product="Dell XPS 13",
        response=ProductQuestionsResponse(
            FinalResponse="Yes, the Dell XPS 13 is fully compatible with Windows 11 and comes with Windows 11 pre-installed on newer models.",
            QuestionsAnswered="",
            QuestionsToEscalate="",
            Confidence=0.98,
        ),
        testcase="dell_xps_13_windows_11_compatibility",
    ),
    ProductQuestionsData(
        query="What's the battery life of the Sony WH-1000XM4?",
        product="Sony WH-1000XM4",
        response=ProductQuestionsResponse(
            FinalResponse="The Sony WH-1000XM4 offers up to 30 hours of battery life with noise canceling on, and up to 38 hours with noise canceling off.",
            QuestionsAnswered="",
            QuestionsToEscalate="",
            Confidence=0.92,
        ),
        testcase="sony_wh_1000xm4_battery_life",
    ),
    ProductQuestionsData(
        query="Does the iPad Pro 2021 come with a charger?",
        product="iPad Pro 2021",
        response=ProductQuestionsResponse(
            FinalResponse="Yes, the iPad Pro 2021 comes with a 20W USB-C power adapter and a USB-C charging cable in the box.",
            QuestionsAnswered="",
            QuestionsToEscalate="",
            Confidence=0.96,
        ),
        testcase="ipad_pro_2021_charger",
    ),
    ProductQuestionsData(
        query="What's the difference between Canon EOS R5 and R6?",
        products=["Canon EOS R5", "Canon EOS R6"],
        response=ProductQuestionsResponse(
            FinalResponse="The main differences are: 1) R5 has 45MP vs R6's 20MP sensor, 2) R5 shoots 8K video vs R6's 4K, 3) R5 has higher resolution EVF, 4) R5 has CFexpress + SD card slots vs R6's dual SD slots, 5) R5 is more expensive",
            QuestionsAnswered="",
            QuestionsToEscalate="",
            Confidence=0.94,
        ),
        testcase="canon_eos_r5_vs_r6",
    ),
    ProductQuestionsData(
        query="Is the Nintendo Switch OLED waterproof?",
        product="Nintendo Switch OLED",
        response=ProductQuestionsResponse(
            FinalResponse="No, the Nintendo Switch OLED is not waterproof and should be kept away from water to prevent damage.",
            QuestionsAnswered="",
            QuestionsToEscalate="",
            Confidence=1.0,
        ),
        testcase="nintendo_switch_oled_waterproof",
    ),
    ProductQuestionsData(
        query="What type of memory cards does the GoPro Hero 10 accept?",
        product="GoPro Hero 10",
        response=ProductQuestionsResponse(
            FinalResponse="The GoPro Hero 10 accepts microSD, microSDHC, and microSDXC cards with a minimum class rating of UHS-I V30.",
            QuestionsAnswered="",
            QuestionsToEscalate="",
            Confidence=0.93,
        ),
        testcase="gopro_hero_10_memory_cards",
    ),
    ProductQuestionsData(
        query="Does the Microsoft Surface Laptop 4 have a touchscreen?",
        product="Microsoft Surface Laptop 4",
        response=ProductQuestionsResponse(
            FinalResponse="Yes, all Microsoft Surface Laptop 4 models come with a 13.5-inch or 15-inch PixelSense touchscreen display.",
            QuestionsAnswered="",
            QuestionsToEscalate="",
            Confidence=0.97,
        ),
        testcase="microsoft_surface_laptop_4_touchscreen",
    ),
    ProductQuestionsData(
        query="What's the maximum zoom on the Nikon P1000?",
        product="Nikon P1000",
        response=ProductQuestionsResponse(
            FinalResponse="The Nikon P1000 features a 125x optical zoom range, equivalent to 24-3000mm in 35mm format, making it one of the highest zoom ranges available in a compact camera.",
            QuestionsAnswered="",
            QuestionsToEscalate="",
            Confidence=0.91,
        ),
        testcase="nikon_p1000_maximum_zoom",
    ),
]

# Per-execution retry counters keyed by thread_id — safe for parallel runs.
_retry_counters: dict[str, dict[str, int]] = {}

# Define a type for our handler functions
HandlerFunction = Callable[[str, str, str | None, int], ProductQuestionsResponse]


def get_macbook_response(
    query: str, product: str, manufacturer: str | None, attempt: int
) -> ProductQuestionsResponse:
    """Handler for MacBook Pro M2 queries with generic error simulation."""
    if attempt == 1:
        raise Exception("Temporary error fetching product information")
    return ProductQuestionsResponse(
        FinalResponse="The MacBook Pro M2 offers up to 22 hours of battery life, the longest of any Mac ever. This includes up to 22 hours of Apple TV app movie playback and up to 18 hours of wireless web browsing.",
        QuestionsAnswered=query,
        QuestionsToEscalate="",
        Confidence=0.95,
    )


def get_samsung_tv_response(
    query: str, product: str, manufacturer: str | None, attempt: int
) -> ProductQuestionsResponse:
    """Handler for Samsung Neo QLED TV queries with timeout error simulation."""
    if attempt == 1:
        raise TimeoutError("Network timeout when connecting to product database")
    return ProductQuestionsResponse(
        FinalResponse="The Samsung Neo QLED TV supports HDR10+ and comes with a built-in smart hub that allows you to access streaming services without additional devices. It also features Quantum Matrix Technology for enhanced contrast and brightness control.",
        QuestionsAnswered=query,
        QuestionsToEscalate="",
        Confidence=0.92,
    )


def get_sony_camera_response(
    query: str, product: str, manufacturer: str | None, attempt: int
) -> ProductQuestionsResponse:
    """Handler for Sony Alpha Camera queries with connection error simulation."""
    if attempt == 1:
        raise ConnectionError("Database connection failed when retrieving product specs")
    return ProductQuestionsResponse(
        FinalResponse="The Sony Alpha Camera series offers full-frame sensors with excellent low-light performance. Most models have in-body image stabilization and can record 4K video. Their autofocus systems are known for reliability and speed, especially for eye-tracking.",
        QuestionsAnswered=query,
        QuestionsToEscalate="",
        Confidence=0.89,
    )


def get_nvidia_gpu_response(
    query: str, product: str, manufacturer: str | None, attempt: int
) -> ProductQuestionsResponse:
    """Handler for NVIDIA RTX 4090 queries with permission error simulation."""
    if attempt == 1:
        raise PermissionError("Authentication failed when accessing product information")
    return ProductQuestionsResponse(
        FinalResponse="The NVIDIA RTX 4090 is the flagship GPU with 24GB GDDR6X memory and support for DLSS 3.0 and ray tracing. It's ideal for 4K gaming and demanding creative workloads like 3D rendering and AI processing.",
        QuestionsAnswered=query,
        QuestionsToEscalate="",
        Confidence=0.93,
    )


def get_ipad_response(
    query: str, product: str, manufacturer: str | None, attempt: int
) -> ProductQuestionsResponse:
    """Handler for iPad Pro 2023 queries with rate limit error simulation."""
    if attempt == 1:
        raise RuntimeError("Rate limit exceeded for product API requests")
    return ProductQuestionsResponse(
        FinalResponse="The iPad Pro 2023 features the M2 chip, providing excellent performance for creative professionals. It supports the Apple Pencil 2 with hover functionality and comes with a Liquid Retina XDR display in the 12.9-inch model for superior color accuracy and HDR content viewing.",
        QuestionsAnswered=query,
        QuestionsToEscalate="",
        Confidence=0.97,
    )


def get_bose_headphones_response(
    query: str, product: str, manufacturer: str | None, attempt: int
) -> ProductQuestionsResponse:
    """Handler for Bose QuietComfort Ultra queries with incomplete data simulation."""
    if attempt == 1:
        return ProductQuestionsResponse(
            FinalResponse="",
            QuestionsAnswered="",
            QuestionsToEscalate="Error: Incomplete product data retrieved. Please try again.",
            Confidence=0.1,
        )
    return ProductQuestionsResponse(
        FinalResponse="The Bose QuietComfort Ultra headphones feature advanced noise cancellation technology, spatial audio support, and up to 24 hours of battery life. They come with both touch controls and physical buttons, and support multipoint Bluetooth connection to pair with two devices simultaneously.",
        QuestionsAnswered=query,
        QuestionsToEscalate="",
        Confidence=0.94,
    )


def get_regular_product_response(
    query: str, product: str, manufacturer: str | None, attempt: int
) -> ProductQuestionsResponse:
    """Handler for regular product queries using the product_questions_data."""
    for item in product_questions_data:
        # Check for exact query match
        if item.query == query:
            response = item.response.model_copy()
            response.QuestionsAnswered = query
            return response

        # Check for product match
        if item.product and item.product in product:
            response = item.response.model_copy()
            response.QuestionsAnswered = query
            return response

    # If no match found, return a default response
    return ProductQuestionsResponse(
        FinalResponse="I don't have specific information about this product. Please contact our customer service for more details.",
        QuestionsAnswered="",
        QuestionsToEscalate=query,
        Confidence=0.3,
    )


# Map of product names to their handler functions
product_handlers: dict[str, HandlerFunction] = {
    "MacBook Pro M2": get_macbook_response,
    "Samsung Neo QLED TV": get_samsung_tv_response,
    "Sony Alpha Camera": get_sony_camera_response,
    "NVIDIA RTX 4090": get_nvidia_gpu_response,
    "iPad Pro 2023": get_ipad_response,
    "Bose QuietComfort Ultra": get_bose_headphones_response,
}


def get_product_response(
    in_CustomerQuery: str,  # noqa: N803
    in_ProductName: str,  # noqa: N803
    in_ProductManufacturer: str | None = None,  # noqa: N803
) -> ProductQuestionsResponse:
    """
    Get response for product questions based on the query and product information.

    Args:
        in_CustomerQuery (str): Customer's question
        in_ProductName (str): Product name extracted from the query
        in_ProductManufacturer (Optional[str]): Product manufacturer if available

    Returns:
        ProductQuestionsResponse: Response with information about the product
    """
    thread_id = ensure_config().get("configurable", {}).get("thread_id", "default")
    counter = _retry_counters.setdefault(thread_id, {})

    # Find the handler for this product if it exists
    for product_key in product_handlers:
        if product_key in in_ProductName:
            counter[product_key] = counter.get(product_key, 0) + 1
            return product_handlers[product_key](
                in_CustomerQuery, in_ProductName, in_ProductManufacturer, counter[product_key]
            )

    # Default to regular product response for products without special handling
    return get_regular_product_response(in_CustomerQuery, in_ProductName, in_ProductManufacturer, 1)
