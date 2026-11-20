"""LLM call costs: price lookup (Bedrock region prefixes included), gateway costs, driver totals."""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agents_gym_core.models.model_spec import ModelSpec
from agents_gym_core.runtime.cost_tracking import (
    ASSOCIATED_DOLLAR_COST_KEY,
    DRIVER_TAG,
    UNSUPPORTED_MODEL_COST,
    CostRecord,
    CostTotals,
    CostTrackingCallback,
    _extract_reported_cost,
    compute_cost,
    load_cost_record,
)
from agents_gym_core.runtime.runners.result_aggregator import (
    _merge_cost_record_into,
)


class TestBedrockRegionPrefixLookup:
    def test_eu_prefixed_anthropic_model_is_priced(self):
        unprefixed = compute_cost("anthropic.claude-sonnet-4-6", 1000, 1000, 0)
        prefixed = compute_cost("eu.anthropic.claude-sonnet-4-6", 1000, 1000, 0)

        assert prefixed != UNSUPPORTED_MODEL_COST
        assert prefixed == unprefixed

    def test_us_and_apac_prefixes_also_resolve(self):
        base = compute_cost("anthropic.claude-sonnet-4-6", 500, 500, 0)

        assert compute_cost("us.anthropic.claude-sonnet-4-6", 500, 500, 0) == base
        assert compute_cost("apac.anthropic.claude-sonnet-4-6", 500, 500, 0) == base

    def test_unknown_model_still_unpriced(self):
        assert compute_cost("eu.anthropic.made-up-model", 1000, 1000, 0) == UNSUPPORTED_MODEL_COST
        assert compute_cost("not-a-real-model", 1000, 1000, 0) == UNSUPPORTED_MODEL_COST


class TestGatewayReportedCost:
    """A routed request's model slot holds the policy name, which the price
    table cannot price."""

    def test_reported_cost_wins_over_the_price_table(self):
        record = CostRecord()

        record.add_usage(
            model_name="gpt-4o-2024-11-20", input_tokens=1000, output_tokens=1000, cached_tokens=0
        )
        table_only = record.total_cost_usd
        record.add_usage(
            model_name="gpt-4o-2024-11-20",
            input_tokens=1000,
            output_tokens=1000,
            cached_tokens=0,
            reported_cost=0.5,
        )

        assert record.total_cost_usd == round(table_only + 0.5, 8)
        assert record.gateway_priced_calls == 1
        assert record.unpriced_calls == 0

    def test_a_policy_name_is_priced_only_when_the_gateway_reports_it(self):
        unpriced = CostRecord()
        unpriced.add_usage(
            model_name="random", input_tokens=100, output_tokens=100, cached_tokens=0
        )

        assert unpriced.total_cost_usd == 0.0
        assert unpriced.unpriced_calls == 1
        assert unpriced.by_model["random"].num_calls == 1

        priced = CostRecord()
        priced.add_usage(
            model_name="random",
            input_tokens=100,
            output_tokens=100,
            cached_tokens=0,
            reported_cost=0.125,
        )

        assert priced.total_cost_usd == 0.125
        assert priced.by_model["random"].cost_usd == 0.125
        assert priced.unpriced_calls == 0

    def test_reported_cost_is_read_from_response_metadata(self):
        message = AIMessage(
            content="hi",
            response_metadata={ASSOCIATED_DOLLAR_COST_KEY: 0.0042, "model_name": "gpt-5.6-sol"},
        )

        assert _extract_reported_cost(message) == 0.0042

    def test_a_missing_or_non_numeric_cost_falls_back_to_the_table(self):
        assert _extract_reported_cost(AIMessage(content="hi")) is None
        assert (
            _extract_reported_cost(
                AIMessage(content="hi", response_metadata={ASSOCIATED_DOLLAR_COST_KEY: "0.5"})
            )
            is None
        )
        # bool is an int subclass; True must not become a $1 call.
        assert (
            _extract_reported_cost(
                AIMessage(content="hi", response_metadata={ASSOCIATED_DOLLAR_COST_KEY: True})
            )
            is None
        )


class TestCallbackRecordsGatewayOnlyCost:
    """Both OpenAI adapters can answer successfully without a usage block."""

    @staticmethod
    def _record_for(message: AIMessage) -> CostRecord:
        callback = CostTrackingCallback()
        asyncio.run(
            callback.on_llm_end(
                LLMResult(generations=[[ChatGeneration(message=message)]]),
                run_id=uuid4(),
            )
        )
        return callback.record

    def test_reported_cost_without_usage_metadata_is_still_recorded(self):
        record = self._record_for(
            AIMessage(
                content="hi",
                response_metadata={
                    ASSOCIATED_DOLLAR_COST_KEY: 0.0042,
                    "model_name": "gpt-5.6-sol",
                },
            )
        )

        assert record.total_cost_usd == 0.0042
        assert record.num_llm_calls == 1
        assert record.gateway_priced_calls == 1
        assert record.by_model["gpt-5.6-sol"].cost_usd == 0.0042
        assert record.by_model["gpt-5.6-sol"].input_tokens == 0

    def test_usage_metadata_still_drives_tokens_when_both_are_present(self):
        record = self._record_for(
            AIMessage(
                content="hi",
                response_metadata={
                    ASSOCIATED_DOLLAR_COST_KEY: 0.5,
                    "model_name": "gpt-4o-2024-11-20",
                },
                usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
            )
        )

        assert record.input_tokens == 10
        assert record.output_tokens == 3
        assert record.total_cost_usd == 0.5

    def test_a_message_with_neither_usage_nor_cost_is_skipped(self):
        record = self._record_for(AIMessage(content="hi", response_metadata={"model_name": "x"}))

        assert record.num_llm_calls == 0
        assert record.total_cost_usd == 0.0


class TestDriverCallsAreTrackedSeparately:
    """Tools that build their own chat model call the gateway inside the same
    run; only the loop's model carries ``DRIVER_TAG``."""

    @staticmethod
    def _message(model: str, cost: float) -> AIMessage:
        return AIMessage(
            content="hi",
            response_metadata={ASSOCIATED_DOLLAR_COST_KEY: cost, "model_name": model},
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )

    def test_only_tagged_calls_land_in_the_driver_block(self):
        callback = CostTrackingCallback()
        for model, cost, tags in (
            ("gpt-5.6-luna", 0.001, [DRIVER_TAG]),
            ("gpt-5.6-sol", 0.02, [DRIVER_TAG, "seq:step:2"]),
            ("gpt-4o-2024-11-20", 0.5, ["tool"]),
            ("gpt-4o-2024-11-20", 0.5, None),
        ):
            asyncio.run(
                callback.on_llm_end(
                    LLMResult(generations=[[ChatGeneration(message=self._message(model, cost))]]),
                    run_id=uuid4(),
                    tags=tags,
                )
            )
        record = callback.record

        assert record.num_llm_calls == 4
        assert record.total_cost_usd == pytest.approx(1.021)
        assert set(record.by_model) == {"gpt-5.6-luna", "gpt-5.6-sol", "gpt-4o-2024-11-20"}

        assert record.driver is not None
        assert record.driver.num_llm_calls == 2
        assert record.driver.total_cost_usd == pytest.approx(0.021)
        assert record.driver.gateway_priced_calls == 2
        assert set(record.driver.by_model) == {"gpt-5.6-luna", "gpt-5.6-sol"}

    def test_the_tag_given_to_the_model_spec_reaches_the_callback(self):
        """The driver split rests on ``get_chat_model(..., tags=[DRIVER_TAG])``
        surfacing the tag in ``on_llm_end``; if that link broke, every driver
        total would silently read as zero."""

        def spec(name: str) -> ModelSpec:
            reply = AIMessage(
                content="ok",
                usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                response_metadata={"model_name": name},
            )
            return ModelSpec(name, GenericFakeChatModel, default_config={"messages": iter([reply])})

        driver = spec("driver-model").get_chat_model(tags=[DRIVER_TAG])
        tool = spec("tool-model").get_chat_model()
        callback = CostTrackingCallback()

        asyncio.run(driver.ainvoke("q", config={"callbacks": [callback]}))
        asyncio.run(tool.ainvoke("q", config={"callbacks": [callback]}))

        assert set(callback.record.by_model) == {"driver-model", "tool-model"}
        assert callback.record.driver is not None
        assert set(callback.record.driver.by_model) == {"driver-model"}
        assert callback.record.driver.num_llm_calls == 1

    def test_a_record_without_a_driver_block_loads_with_driver_none(self, tmp_path: Path):
        (tmp_path / "cost.json").write_text(
            json.dumps({"total_cost_usd": 1.0, "num_llm_calls": 2, "by_model": {}})
        )
        record = load_cost_record(tmp_path)

        assert record is not None
        assert record.driver is None
        assert record.num_llm_calls == 2


class TestRoutedDrawsAttributeToTheDrawnCandidate:
    """A routing policy draws per request, so one run's calls land on several
    candidates and a candidate can come up more than once."""

    @staticmethod
    def _record_for(model_names: list[str]) -> CostRecord:
        callback = CostTrackingCallback()
        for index, model_name in enumerate(model_names):
            asyncio.run(
                callback.on_llm_end(
                    LLMResult(
                        generations=[
                            [
                                ChatGeneration(
                                    message=AIMessage(
                                        content="hi",
                                        response_metadata={
                                            ASSOCIATED_DOLLAR_COST_KEY: 0.1,
                                            "model_name": model_name,
                                        },
                                        usage_metadata={
                                            "input_tokens": 10,
                                            "output_tokens": index + 1,
                                            "total_tokens": 11 + index,
                                        },
                                    )
                                )
                            ]
                        ],
                    ),
                    run_id=uuid4(),
                )
            )
        return callback.record

    def test_a_candidate_drawn_twice_accumulates_under_one_key(self):
        record = self._record_for(["gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-sol"])

        assert set(record.by_model) == {"gpt-5.6-sol", "gpt-5.6-luna"}
        assert record.by_model["gpt-5.6-sol"].num_calls == 2
        assert record.by_model["gpt-5.6-sol"].input_tokens == 20
        assert record.by_model["gpt-5.6-sol"].output_tokens == 4
        assert record.by_model["gpt-5.6-sol"].cost_usd == 0.2
        assert record.by_model["gpt-5.6-luna"].num_calls == 1
        assert record.num_llm_calls == 3
        assert record.total_cost_usd == round(0.3, 8)


class TestGatewayPricingCoverageSurvivesAggregation:
    """``gateway_priced_calls`` is the only signal separating reported dollars
    from locally estimated ones."""

    def test_merging_run_records_sums_gateway_priced_calls(self):
        first = CostRecord()
        first.add_usage(
            model_name="gpt-5.6-sol",
            input_tokens=10,
            output_tokens=10,
            cached_tokens=0,
            reported_cost=0.25,
        )
        second = CostRecord()
        second.add_usage(
            model_name="gpt-5.6-luna",
            input_tokens=10,
            output_tokens=10,
            cached_tokens=0,
            reported_cost=0.75,
        )
        second.add_usage(
            model_name="not-a-real-model", input_tokens=10, output_tokens=10, cached_tokens=0
        )

        _merge_cost_record_into(first, second)

        assert first.gateway_priced_calls == 2
        assert first.unpriced_calls == 1
        assert first.total_cost_usd == 1.0

    def test_merging_run_records_sums_the_driver_block(self):
        first, second = CostRecord(driver=CostTotals()), CostRecord(driver=CostTotals())
        first.add_usage("gpt-5.6-sol", 10, 10, 0, reported_cost=0.25, driver=True)
        second.add_usage("gpt-5.6-luna", 10, 10, 0, reported_cost=0.75, driver=True)
        second.add_usage("gpt-4o-2024-11-20", 10, 10, 0, reported_cost=0.5)

        _merge_cost_record_into(first, second)

        assert first.total_cost_usd == 1.5
        assert first.driver is not None
        assert first.driver.total_cost_usd == 1.0
        assert set(first.driver.by_model) == {"gpt-5.6-sol", "gpt-5.6-luna"}

    def test_merging_a_record_without_a_driver_block_drops_the_block(self):
        tracked = CostRecord(driver=CostTotals())
        tracked.add_usage("gpt-5.6-sol", 10, 10, 0, reported_cost=0.25, driver=True)
        legacy = CostRecord()
        legacy.add_usage("gpt-5.6-luna", 10, 10, 0, reported_cost=0.75)

        _merge_cost_record_into(tracked, legacy)

        assert tracked.total_cost_usd == 1.0
        assert tracked.driver is None


def test_an_optimization_run_sums_its_sessions_and_its_rollouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agents_gym_core.prompt_optimization import run_accounting
    from agents_gym_core.prompt_optimization.mcp_tools.helpers import record_started_experiment

    workspace, experiments = tmp_path / "tmp_ws", tmp_path / "experiments"
    monkeypatch.setenv("COACH_OUTPUT_DIR", str(workspace))
    monkeypatch.setattr(run_accounting, "experiments_dir", lambda: experiments)

    def usage(calls: int) -> dict:
        totals = CostTotals()
        for _ in range(calls):
            totals.add_usage("gpt-4.1-mini-2025-04-14", 1000, 100, 0)
        return totals.model_dump()

    sessions = workspace / run_accounting.SESSIONS_DIRNAME
    sessions.mkdir(parents=True)
    for name, calls in (("a", 2), ("b", 3)):  # a first session and its -c continuation
        (sessions / f"{name}.json").write_text(json.dumps({"optimizer_cost": usage(calls)}))
    for name, calls in (("run_round_000_001", 4), ("run_round_001_001", 1), ("other_eval", 50)):
        datapoint = experiments / name / "agent" / "train" / "basic" / "dp" / "run_1"
        datapoint.mkdir(parents=True)
        (datapoint / "cost.json").write_text(json.dumps(usage(calls)))
    record_started_experiment("run_round_000_001")
    record_started_experiment("run_round_001_001")

    (sessions / "c.json").write_text('{"optimizer_cost": {"input_')  # cut off mid-write
    bad = experiments / "run_round_001_001" / "agent" / "train" / "basic" / "dp2" / "run_1"
    bad.mkdir(parents=True)
    (bad / "cost.json").write_text("")
    assert run_accounting.optimizer_cost(workspace)["num_llm_calls"] == 5
    rollouts = run_accounting.eval_cost()  # only the experiments this run started
    assert (rollouts["num_llm_calls"], rollouts["input_tokens"]) == (5, 5000)
    assert rollouts["total_cost_usd"] == pytest.approx(5 * usage(1)["total_cost_usd"])


def test_each_round_records_its_evaluation_runs_time_and_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import time

    from agents_gym_core.prompt_optimization.mcp_tools import tools_runner

    experiments = tmp_path / "experiments"
    monkeypatch.setattr(tools_runner, "experiments_dir", lambda: experiments)
    monkeypatch.setenv("COACH_OUTPUT_DIR", str(tmp_path / "tmp_ws"))
    now = time.time()
    for n, (calls, seconds) in enumerate(((2, 40.0), (3, 60.0)), start=1):
        name = f"react_opt_x_round_001_00{n}"
        datapoint = experiments / name / "agent" / "train" / "basic" / "dp" / "run_1"
        datapoint.mkdir(parents=True)
        totals = CostTotals()
        for _ in range(calls):
            totals.add_usage("gpt-4.1-mini-2025-04-14", 1000, 100, 0)
        (datapoint / "cost.json").write_text(json.dumps(totals.model_dump()))
        scores = experiments / name / "final_score_per_agent.json"
        scores.write_text("{}")
        os.utime(scores, (now, now))
        run_info = {"experiment_name": name, "model": "m", "started_at": now - seconds}
        tools_runner._write_round_stats(run_info, "agent:train", 10, 0.5)

    stats = json.loads((tmp_path / "tmp_ws" / "round_001" / "round_stats.json").read_text())
    assert [r["duration_s"] for r in stats["runs"]] == [40.0, 60.0]
    assert stats["eval_duration_s"] == 100.0
    one_call = compute_cost("gpt-4.1-mini-2025-04-14", 1000, 100)
    assert stats["eval_cost_usd"] == pytest.approx(5 * one_call)


def test_run_timing_sums_sessions_and_finds_the_best_round(tmp_path: Path) -> None:
    from agents_gym_core.prompt_optimization import run_accounting

    sessions = tmp_path / run_accounting.SESSIONS_DIRNAME
    sessions.mkdir()
    for name, started, duration in (
        ("a", "2026-10-20T10:00:00+00:00", 600.0),
        ("b", "2026-10-20T12:00:00+00:00", 300.0),
    ):
        (sessions / f"{name}.json").write_text(
            json.dumps({"started_at": started, "duration_s": duration})
        )
    for n, (score, finished) in enumerate(
        (
            (0.5, "2026-10-20T10:05:00+00:00"),
            (0.7, "2026-10-20T12:03:00+00:00"),
            (0.7, "2026-10-20T12:04:00+00:00"),
        )
    ):
        round_dir = tmp_path / f"round_{n:03d}"
        round_dir.mkdir()
        (round_dir / "round_stats.json").write_text(
            json.dumps(
                {"round": round_dir.name, "score": score, "runs": [{"finished_at": finished}]}
            )
        )

    timing = run_accounting.timing(tmp_path)
    assert timing == {
        "total_duration_s": 900.0,
        "best_round": "round_001",  # the first of the equally good rounds
        "best_score": 0.7,
        "time_to_best_round_s": 2 * 3600 + 3 * 60,  # from the first session's start
    }


@pytest.mark.parametrize(
    ("model", "threshold"),
    [("gpt-5.4-2026-03-05", 272_000), ("claude-haiku-5-5", 100_000), ("gemini-2.5-pro", 200_000)],
)
def test_long_prompts_switch_to_the_higher_rate_above_the_threshold(
    model: str, threshold: int
) -> None:
    def per_token(n: int) -> float:
        return compute_cost(model, n, 0) / n

    assert per_token(threshold) < per_token(threshold + 1)


def test_an_announced_price_change_applies_from_its_date(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime

    import agents_gym_core.runtime.cost_tracking as cost_tracking

    class Clock:
        now_value = datetime(2026, 11, 30, tzinfo=UTC)

        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return cls.now_value

    monkeypatch.setattr(cost_tracking, "datetime", Clock)
    before = compute_cost("gemini-4-argon", 1_000_000, 0)
    Clock.now_value = datetime(2026, 12, 1, tzinfo=UTC)
    assert (before, compute_cost("gemini-4-argon", 1_000_000, 0)) == (2.0, 4.0)


def test_the_summary_leaves_out_a_best_round_time_it_does_not_have(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agents_gym_core.prompt_optimization import run_accounting

    totals = {
        "total_duration_s": 120.0,
        "best_round": "round_001",
        "optimizer_cost": CostTotals().model_dump(),
        "eval_cost": CostTotals().model_dump(),
        "total_cost_usd": 0.0,
    }
    run_accounting.print_run_summary(totals)
    out = capsys.readouterr().out
    assert "Time:      120s\n" in out and "after" not in out


def test_a_dated_snapshot_is_priced_through_its_dateless_name() -> None:
    dated, dateless = (
        compute_cost("gpt-6.1-sol-2026-09-29", 1000, 100),
        compute_cost("gpt-6.1-sol", 1000, 100),
    )

    assert dated == dateless != UNSUPPORTED_MODEL_COST
    assert compute_cost("not-a-real-model-2026-01-01", 1000, 100) == UNSUPPORTED_MODEL_COST
