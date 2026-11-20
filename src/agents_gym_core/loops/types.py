"""Type definitions for agent loops."""

from enum import StrEnum

from pydantic import BaseModel, Field
from uipath_langchain.agent.react.types import AgentGraphNode


class LoopName(StrEnum):
    """Available loops, selected as ``name`` or ``name:version`` (e.g. ``basic:any``).

    A tool-calling loop's versions are named after their behaviour; ``auto`` (the model
    chooses whether to call a tool) is the default. ``deep-agents`` has only ``auto``;
    ``dspy-program``'s version is the dotted path of the DSPy program it runs.
    """

    BASIC = "basic"
    PLANNING = "planning"
    VALIDATION = "validation"
    SUMMARIZING = "summarizing"
    DEEP_AGENTS = "deep-agents"
    DSPY_PROGRAM = "dspy-program"


DEFAULT_LOOP_VERSION = "auto"


class LoopDetails(BaseModel):
    """Details about a loop configuration.

    Contains loop identification and model configuration.
    Models is a mapping from model type (e.g., "model", "planning_llm") to model name.
    """

    # LoopName is a StrEnum (subclass of str), so plain str accepts both
    # registered enum values and arbitrary dotted-path loop identifiers.
    loop_name: str
    loop_version: str = DEFAULT_LOOP_VERSION
    models: dict[str, str] = Field(default_factory=dict)

    @property
    def loop_key(self) -> str:
        """Construct the loop key string.

        Format: "{loop_name}__{loop_version}__{key}:{value},..."
        Example: "basic__auto__model:gpt-4o-2024-11-20"
        Example: "basic__auto__model:gpt-4o-2024-11-20,embeddings:text-embedding-3-large"
        """
        base = f"{self.loop_name}__{self.loop_version}"

        if not self.models:
            return base

        middle = sorted(k for k in self.models if k not in ("model", "embeddings"))
        ordered_keys = (
            (["model"] if "model" in self.models else [])
            + middle
            + (["embeddings"] if "embeddings" in self.models else [])
        )
        model_str = ",".join(f"{k}:{self.models[k]}" for k in ordered_keys)
        return f"{base}__{model_str}"

    def with_model(self, model_type: str, model_name: str) -> "LoopDetails":
        """Create a new LoopDetails with an added model.

        Args:
            model_type: Type identifier for the model (e.g., "model", "planning_llm")
            model_name: Name of the model (e.g., "gpt-4o-2024-11-20")

        Returns:
            New LoopDetails with the model added
        """
        new_models = {**self.models, model_type: model_name}
        return self.model_copy(update={"models": new_models})


class ExtendedGraphNode(StrEnum):
    """Extended graph node names for all loop types.

    Extends the base AgentGraphNode with additional node types
    used across different loop implementations.
    """

    INIT = AgentGraphNode.INIT
    AGENT = AgentGraphNode.AGENT
    LLM = AgentGraphNode.LLM
    TOOLS = AgentGraphNode.TOOLS
    TERMINATE = AgentGraphNode.TERMINATE
    PLANNING = "planning"
    VALIDATION = "validation"
    SUMMARIZING = "summarizing"
