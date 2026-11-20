"""IFBench's scorer needs NLTK data; a fresh machine without it must fail loudly, not per datapoint."""

import nltk
import pytest

from agents_gym_core import gepa_artifact
from agents_gym_core.agents.ifbench.init import EVALUATOR_PY


def _find_all_but(missing: str):
    def find(resource: str, *args: object, **kwargs: object) -> str:
        if resource == gepa_artifact.IFBENCH_NLTK_RESOURCES[missing]:
            raise LookupError(resource)
        return resource

    return find


def test_reports_the_missing_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nltk.data, "find", _find_all_but("stopwords"))
    assert gepa_artifact.missing_ifbench_nltk_data() == ["stopwords"]


def test_the_evaluator_refuses_to_load_without_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nltk.data, "find", _find_all_but("punkt_tab"))
    monkeypatch.setattr(nltk, "download", nltk.download)  # the evaluator stubs it out on load
    with pytest.raises(RuntimeError, match="gym init ifbench"):
        exec(compile(EVALUATOR_PY, "ifbench_instruction_following_evaluator.py", "exec"), {})
