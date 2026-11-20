from typing import Annotated

from langchain_core.tools import tool


@tool
async def process_features(
    list_of_features: Annotated[
        list[str], "The list of feature strings from the product's Attributes.Features array"
    ],
) -> str:
    """
    Process a list of features in a string format
    """
    return " ".join(list_of_features)


@tool
async def save_to_cloud(
    product_name: Annotated[str, "The exact product name as given in the input"],
    processed_string: Annotated[str, "The processed features string returned by process_features"],
) -> str:
    """
    Save the product name and processed string output to the cloud
    """
    return f"Saved {product_name}"
