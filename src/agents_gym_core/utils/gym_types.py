from typing import Any

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from uipath.eval.models.evaluation_set import EvaluationSet


class EvaluatorDatapointResult(BaseModel):
    """Results for a single datapoint evaluation."""

    experiment_unique_identifier: str
    agent: str
    agent_version: str
    loop_name: str
    loop_version: str
    loop_key: str
    loop_models: dict[str, str]
    datapoint_name: str
    score: float
    evaluator_id: str
    agent_output: Any
    evaluation_criteria: Any
    details: Any
    evaluator_weights: dict[str, float] = Field(default_factory=dict)
    judge_model: str | None = None
    judge_prompt: str | None = None
    run_number: int = 1
    split: str = "all_data"


class Agent(BaseModel):
    """Agent configuration with prompts, schemas, and tools."""

    name: str
    system_prompt: str
    user_prompt: str
    evaluation_set: EvaluationSet
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    tools: list[BaseTool] = Field(default_factory=list)
    version: str = "base"
    evaluator_weights: dict[str, float] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    datapoint_splits: dict[str, str] = Field(default_factory=dict)
    """Datapoint name to split label (e.g. ``"train"``, ``"val"``, ``"test"``, ``"all_data"``),
    taken from the eval-set file names."""
    loop: str = Field(default="")
    """Default loop spec for this agent (e.g. ``"dspy-program:my.module.MyProgram"``).
    When set, it replaces the default ``basic`` entry of ``--loops`` for this agent;
    any other loop given in ``--loops`` still runs."""

    signature_prompts: dict[str, str] = Field(default_factory=dict)
    """DSPy predictor name to instruction text, which replaces that predictor's signature
    instructions in the dspy-program loop. From ``agent.signature_prompts``; values are
    ``core::`` file references or inline strings."""

    signature_demos: dict[str, list[dict]] = Field(default_factory=dict)
    """DSPy predictor name to few-shot demo dicts, which the dspy-program loop sets as that
    predictor's ``demos``. From ``agent.signature_demos``; values are ``core::`` paths to
    JSON files of the form ``{"demos": [...]}``. With demos this reproduces MIPROv2's 'base'
    variant (instructions and bootstrapped few-shots); empty, its 'no-demos' variant."""

    @property
    def is_conversational(self) -> bool:
        """Check if this is a conversational agent based on tags."""
        return "conversational" in self.tags
