"""strictly_typed_pandas schemas for the DataFrames used in reports."""

from dataclasses import dataclass, field


@dataclass
class AgentScoreSummarySchema:
    """One row per (loop, agent, agent_version): mean score across runs and its CI."""

    model: str
    loop_name: str
    loop_version: str
    loop_key: str
    agent: str
    agent_version: str
    score: str  # "mean ± error_margin"
    mean: float
    # Names follow BootstrapStatistics and matplotlib.bxp.
    cilo: float
    cihi: float
    whislo: float  # data min
    whishi: float  # data max


@dataclass
class AgentScorePerRunSchema:
    """One row per run of a (loop, agent, agent_version) combination."""

    model: str
    loop_name: str
    loop_version: str
    loop_key: str
    agent: str
    agent_version: str
    run: str
    score: float


@dataclass
class EvaluatorDatapointResultSchema:
    """Pandas counterpart of ``EvaluatorDatapointResult`` (``utils/gym_types.py``).

    Dict and Any fields are typed ``object``, the dtype pandas stores them as.
    """

    experiment_unique_identifier: str
    agent: str
    datapoint_name: str
    score: float
    evaluator_id: str
    agent_output: object
    evaluation_criteria: object
    details: object
    agent_version: str = "base"
    loop_name: str = "basic"
    loop_version: str = "auto"
    loop_key: str = "basic__auto"
    loop_models: object = field(default_factory=dict)
    split: str = "all_data"
    judge_model: object = None
    judge_prompt: object = None
    run_number: int = 1
    evaluator_weights: object = field(default_factory=dict)
