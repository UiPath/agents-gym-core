"""LLM cost tracking: callback, price table and per-datapoint ``cost.json``.

The callback sums the ``usage_metadata`` of LangChain chat model calls into a
:class:`CostRecord`, saved next to ``output.json`` as ``cost.json``. LangChain's
SQLite cache stores ``AIMessage.usage_metadata`` with the cached generation, so
``on_llm_end`` reports the same usage on cache hits.

A datapoint loaded from disk in a fill mode makes no LLM calls; its saved
``cost.json`` is read instead, so datapoints executed in an earlier run of the
same experiment directory still count toward the per-agent cost.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel, Field

COST_FILENAME = "cost.json"
UNSUPPORTED_MODEL_COST = -1.0
ASSOCIATED_DOLLAR_COST_KEY = "associated_dollar_cost"
DRIVER_TAG = "gym:driver"

# AWS Bedrock cross-region inference profile IDs prefix the model id
# (e.g. "eu.anthropic.claude-sonnet-4-6"); the prefix is stripped before the price lookup.
_BEDROCK_REGION_PREFIXES = ("global.", "eu.", "us.", "apac.")


class MessageCost(BaseModel):
    """Per-million-token pricing for a single model."""

    input_cost: float
    cached_input_cost: float
    output_cost: float


# Prices in USD per million tokens (input, cached input, output), standard tier, from each
# provider's pricing page as of 2026-10-20. Cache-write surcharges are not modelled, so a run
# that writes many cache entries costs somewhat more than computed. Model names differ across
# the direct APIs, Bedrock and Vertex, so a model may be registered under several names.
# Models older than GPT-4o 2024-11-20 and the Claude 4.5 generation are not priced.

_Price = tuple[float, float, float]


def _const_price(
    input_cost: float, cached_input_cost: float, output_cost: float
) -> Callable[[int], MessageCost]:
    def _fn(_: int) -> MessageCost:
        return MessageCost(
            input_cost=input_cost,
            cached_input_cost=cached_input_cost,
            output_cost=output_cost,
        )

    return _fn


def _tiered_price(threshold: int, low: _Price, high: _Price) -> Callable[[int], MessageCost]:
    """``low`` when the request's input is at most ``threshold`` tokens, else ``high``."""

    def _fn(n: int) -> MessageCost:
        t = low if n <= threshold else high
        return MessageCost(input_cost=t[0], cached_input_cost=t[1], output_cost=t[2])

    return _fn


def _dated_price(change: date, before: _Price, after: _Price) -> Callable[[int], MessageCost]:
    """``before`` until the provider's announced price change takes effect, then ``after``."""

    def _fn(_: int) -> MessageCost:
        t = before if datetime.now(UTC).date() < change else after
        return MessageCost(input_cost=t[0], cached_input_cost=t[1], output_cost=t[2])

    return _fn


# OpenAI: developers.openai.com/api/docs/pricing. GPT-5.4 and later charge a higher rate for
# the whole request above 272K input tokens. Models without a cached rate reuse the input price.
_OPENAI_LONG_CONTEXT = 272_000
_OPENAI_PRICES: dict[str, Callable[[int], MessageCost]] = {
    # GPT-4o (the paper's LLM judges)
    "gpt-4o-2024-11-20": _const_price(2.50, 1.25, 10.00),
    # GPT-4.1 family (GPT-4.1 Mini is the paper's system model on the public benchmarks)
    "gpt-4.1-2025-04-14": _const_price(2.00, 0.50, 8.00),
    "gpt-4.1-mini-2025-04-14": _const_price(0.40, 0.10, 1.60),
    "gpt-4.1-nano-2025-04-14": _const_price(0.10, 0.025, 0.40),
    # GPT-5
    "gpt-5-2025-08-07": _const_price(1.25, 0.125, 10.00),
    "gpt-5-mini-2025-08-07": _const_price(0.25, 0.025, 2.00),
    "gpt-5-nano-2025-08-07": _const_price(0.05, 0.005, 0.40),
    "gpt-5.1-2025-11-13": _const_price(1.25, 0.125, 10.00),
    "gpt-5.2-2025-12-11": _const_price(1.75, 0.175, 14.00),
    "gpt-5.2-codex": _const_price(1.75, 0.175, 14.00),
    "gpt-5.3-codex": _const_price(1.75, 0.175, 14.00),
    # GPT-5.4
    "gpt-5.4-2026-03-05": _tiered_price(
        _OPENAI_LONG_CONTEXT, (2.50, 0.25, 15.00), (5.00, 0.50, 22.50)
    ),
    "gpt-5.4-mini-2026-03-17": _const_price(0.75, 0.075, 4.50),
    "gpt-5.4-nano-2026-03-17": _const_price(0.20, 0.02, 1.25),
    "gpt-5.4-pro-2026-03-05": _tiered_price(
        _OPENAI_LONG_CONTEXT, (30.00, 30.00, 180.00), (60.00, 60.00, 270.00)
    ),
    # GPT-5.5 (OpenAI's snapshot is 2026-04-23; the UiPath LLM Gateway registry names it 2026-04-24)
    "gpt-5.5-2026-04-23": _tiered_price(
        _OPENAI_LONG_CONTEXT, (5.00, 0.50, 30.00), (10.00, 1.00, 45.00)
    ),
    "gpt-5.5-2026-04-24": _tiered_price(
        _OPENAI_LONG_CONTEXT, (5.00, 0.50, 30.00), (10.00, 1.00, 45.00)
    ),
    "gpt-5.5-pro-2026-04-23": _tiered_price(
        _OPENAI_LONG_CONTEXT, (30.00, 30.00, 180.00), (60.00, 60.00, 270.00)
    ),
    # GPT-5.6 (no dated snapshots; the UiPath LLM Gateway registry adds 2026-07-09)
    "gpt-5.6-sol-2026-07-09": _tiered_price(
        _OPENAI_LONG_CONTEXT, (4.00, 0.40, 20.00), (8.00, 0.80, 30.00)
    ),
    "gpt-5.6-terra-2026-07-09": _tiered_price(
        _OPENAI_LONG_CONTEXT, (2.00, 0.20, 12.00), (4.00, 0.40, 18.00)
    ),
    "gpt-5.6-luna-2026-07-09": _tiered_price(
        _OPENAI_LONG_CONTEXT, (0.20, 0.02, 1.20), (0.40, 0.04, 1.80)
    ),
    # GPT-6: the OpenAI API names are dateless; Azure deployments add a snapshot date and are
    # priced through the dateless name (see compute_cost).
    "gpt-6-astra": _tiered_price(_OPENAI_LONG_CONTEXT, (10.00, 1.00, 50.00), (20.00, 2.00, 75.00)),
    "gpt-6-sol": _tiered_price(_OPENAI_LONG_CONTEXT, (2.00, 0.20, 10.00), (4.00, 0.40, 15.00)),
    "gpt-6.1-sol": _tiered_price(_OPENAI_LONG_CONTEXT, (2.00, 0.10, 10.00), (4.00, 0.20, 15.00)),
    "gpt-6-luna": _tiered_price(_OPENAI_LONG_CONTEXT, (0.10, 0.01, 0.50), (0.20, 0.02, 0.75)),
}
# Responses may name a model without its snapshot date.
for _name, _fn in list(_OPENAI_PRICES.items()):
    if (_dateless := re.sub(r"-\d{4}-\d{2}-\d{2}$", "", _name)) != _name:
        _OPENAI_PRICES.setdefault(_dateless, _fn)


# Anthropic: platform.claude.com/docs/en/about-claude/pricing. Cache reads are 0.1x input,
# except Opus 5.5 / Sonnet 5.5 (0.05x) and Fable 5.1 (0.025x). Haiku 5.5 charges more for
# prompts over 100K tokens. Aliases below add the Bedrock (``anthropic.<id>[-v1:0]``) and Vertex
# (``<slug>@<date>``) names.
_ANTHROPIC_CANONICAL: dict[str, Callable[[int], MessageCost]] = {
    # Claude 4.5
    "claude-sonnet-4-5-20250929": _const_price(3.00, 0.30, 15.00),
    "claude-haiku-4-5-20251001": _const_price(1.00, 0.10, 5.00),
    "claude-opus-4-5-20251101": _const_price(5.00, 0.50, 25.00),
    # Claude 4.6 to 4.8 (dateless IDs from here on)
    "claude-sonnet-4-6": _const_price(3.00, 0.30, 15.00),
    "claude-opus-4-6": _const_price(5.00, 0.50, 25.00),
    "claude-opus-4-7": _const_price(5.00, 0.50, 25.00),
    "claude-opus-4-8": _const_price(5.00, 0.50, 25.00),
    # Claude 5
    "claude-sonnet-5": _const_price(2.00, 0.20, 10.00),
    "claude-opus-5": _const_price(5.00, 0.50, 25.00),
    "claude-fable-5": _const_price(10.00, 1.00, 50.00),
    "claude-haiku-5-5": _tiered_price(100_000, (0.10, 0.01, 0.50), (0.50, 0.05, 2.50)),
    "claude-sonnet-5-5": _const_price(2.00, 0.10, 10.00),
    "claude-opus-5-5": _const_price(4.00, 0.20, 20.00),
    "claude-fable-5-1": _const_price(10.00, 0.25, 50.00),
}


def _anthropic_aliases(
    slug: str, price: Callable[[int], MessageCost]
) -> dict[str, Callable[[int], MessageCost]]:
    """The API name plus its Bedrock and Vertex names."""
    aliases = {slug: price, f"anthropic.{slug}-v1:0": price, f"anthropic.{slug}": price}
    # Vertex writes "@date" where the API name has "-date".
    if (split_idx := slug.rfind("-20")) != -1:
        aliases[slug[:split_idx] + "@" + slug[split_idx + 1 :]] = price
    return aliases


_ANTHROPIC_PRICES: dict[str, Callable[[int], MessageCost]] = {}
for _slug, _price in _ANTHROPIC_CANONICAL.items():
    _ANTHROPIC_PRICES.update(_anthropic_aliases(_slug, _price))
# Opus 4.6 is the last Bedrock ID with a "-v1" suffix (the paper's frontier optimizer).
_ANTHROPIC_PRICES["anthropic.claude-opus-4-6-v1"] = _ANTHROPIC_CANONICAL["claude-opus-4-6"]


# Gemini: ai.google.dev/gemini-api/docs/pricing. 2.5-pro and 3.x-pro charge more above 200K
# input tokens; several newer models have announced price increases.
_GEMINI_PRICES: dict[str, Callable[[int], MessageCost]] = {
    "gemini-2.5-flash": _const_price(0.30, 0.03, 2.50),
    "gemini-2.5-flash-lite": _const_price(0.10, 0.01, 0.40),
    "gemini-2.5-pro": _tiered_price(200_000, (1.25, 0.125, 10.00), (2.50, 0.25, 15.00)),
    "gemini-3-flash-preview": _const_price(0.50, 0.05, 3.00),
    "gemini-3-pro-preview": _tiered_price(200_000, (2.00, 0.20, 12.00), (4.00, 0.40, 18.00)),
    "gemini-3.1-pro-preview": _tiered_price(200_000, (2.00, 0.20, 12.00), (4.00, 0.40, 18.00)),
    "gemini-3.1-flash-lite": _const_price(0.25, 0.025, 1.50),
    "gemini-3.5-flash-lite": _const_price(0.30, 0.03, 2.50),
    "gemini-3.6-flash": _dated_price(date(2027, 1, 1), (0.75, 0.075, 3.75), (1.50, 0.15, 7.50)),
    "gemini-3.8-flash": _dated_price(date(2027, 1, 1), (0.75, 0.075, 3.75), (1.50, 0.15, 7.50)),
    "gemini-4-argon": _dated_price(date(2026, 12, 1), (2.00, 0.10, 10.00), (4.00, 0.20, 20.00)),
}


MODEL_COST_PER_MILLION: dict[str, Callable[[int], MessageCost]] = {
    **_OPENAI_PRICES,
    **_ANTHROPIC_PRICES,
    **_GEMINI_PRICES,
}


class ModelUsage(BaseModel):
    """Token usage and cost of one model within one run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    num_calls: int = 0


class CostTotals(BaseModel):
    """Token usage and cost summed over a set of LLM calls."""

    total_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    num_llm_calls: int = 0
    by_model: dict[str, ModelUsage] = Field(default_factory=dict)
    unpriced_calls: int = Field(
        default=0,
        description="Number of LLM calls whose model was not in the price table.",
    )
    gateway_priced_calls: int = Field(
        default=0,
        description="Number of LLM calls costed from the gateway's reported dollar cost.",
    )

    def add_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        reported_cost: float | None = None,
    ) -> None:
        """Add one LLM call.

        The gateway's ``reported_cost`` takes precedence over the price table, which
        cannot price a routing policy name.
        """
        cost = (
            reported_cost
            if reported_cost is not None
            else compute_cost(model_name, input_tokens, output_tokens, cached_tokens)
        )

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cached_tokens += cached_tokens
        self.num_llm_calls += 1

        per_model = self.by_model.setdefault(model_name, ModelUsage())
        per_model.input_tokens += input_tokens
        per_model.output_tokens += output_tokens
        per_model.cached_tokens += cached_tokens
        per_model.num_calls += 1

        if cost == UNSUPPORTED_MODEL_COST:
            self.unpriced_calls += 1
            return
        if reported_cost is not None:
            self.gateway_priced_calls += 1

        self.total_cost_usd = round(self.total_cost_usd + cost, 8)
        per_model.cost_usd = round(per_model.cost_usd + cost, 8)

    def merge(self, other: CostTotals) -> None:
        """Add another total's calls into this one."""
        self.total_cost_usd = round(self.total_cost_usd + other.total_cost_usd, 8)
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens
        self.num_llm_calls += other.num_llm_calls
        self.unpriced_calls += other.unpriced_calls
        self.gateway_priced_calls += other.gateway_priced_calls
        for model_name, usage in other.by_model.items():
            merged = self.by_model.setdefault(model_name, ModelUsage())
            merged.input_tokens += usage.input_tokens
            merged.output_tokens += usage.output_tokens
            merged.cached_tokens += usage.cached_tokens
            merged.num_calls += usage.num_calls
            merged.cost_usd = round(merged.cost_usd + usage.cost_usd, 8)


class CostRecord(CostTotals):
    """Cost of one datapoint run, saved as cost.json.

    ``driver`` covers only the calls of the loop's chat model (tagged ``DRIVER_TAG``);
    calls of models a tool creates count in the totals only. ``None`` means the record
    has no driver breakdown.
    """

    driver: CostTotals | None = None

    def add_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        reported_cost: float | None = None,
        driver: bool = False,
    ) -> None:
        super().add_usage(model_name, input_tokens, output_tokens, cached_tokens, reported_cost)
        if driver:
            if self.driver is None:
                self.driver = CostTotals()
            self.driver.add_usage(
                model_name, input_tokens, output_tokens, cached_tokens, reported_cost
            )


def compute_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """USD cost of one LLM call, or ``UNSUPPORTED_MODEL_COST`` for a model without a price."""
    lookup_name = model_name
    for prefix in _BEDROCK_REGION_PREFIXES:
        if lookup_name.startswith(prefix):
            lookup_name = lookup_name[len(prefix) :]
            break
    price_fn = MODEL_COST_PER_MILLION.get(lookup_name)
    if price_fn is None:  # a dated snapshot of a model priced under its dateless name
        price_fn = MODEL_COST_PER_MILLION.get(re.sub(r"-\d{4}-\d{2}-\d{2}$", "", lookup_name))
    if price_fn is None:
        return UNSUPPORTED_MODEL_COST

    price = price_fn(input_tokens)
    effective_input = max(0, input_tokens - cached_tokens)
    return round(
        (effective_input / 1_000_000) * price.input_cost
        + (cached_tokens / 1_000_000) * price.cached_input_cost
        + (output_tokens / 1_000_000) * price.output_cost,
        8,
    )


def _extract_cached_tokens(message: AIMessage) -> int:
    """Cached-read token count from a message's usage_metadata or response_metadata."""
    usage = dict(message.usage_metadata) if message.usage_metadata else {}
    details = usage.get("input_token_details") if usage else None
    if isinstance(details, dict):
        cached = details.get("cache_read") or details.get("cached_read") or 0
        if cached:
            return int(cached)

    token_usage = (message.response_metadata or {}).get("token_usage") or {}
    return int(token_usage.get("cache_read_input_tokens") or 0)


def _extract_reported_cost(message: AIMessage) -> float | None:
    """The gateway's own dollar cost, present only when the request opted in with
    ``X-UiPath-LlmGateway-IncludeAssociatedDollarCost``."""
    reported = (message.response_metadata or {}).get(ASSOCIATED_DOLLAR_COST_KEY)
    if isinstance(reported, bool) or not isinstance(reported, (int, float)):
        return None
    return float(reported)


def _extract_model_name(message: AIMessage, fallback: str | None) -> str:
    return str(
        (message.response_metadata or {}).get("model_name")
        or (message.response_metadata or {}).get("model")
        or fallback
        or "unknown"
    )


class CostTrackingCallback(AsyncCallbackHandler):
    """Async LangChain callback summing one agent run's LLM usage and cost into :attr:`record`."""

    def __init__(self) -> None:
        super().__init__()
        self.record = CostRecord(driver=CostTotals())

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        is_driver = DRIVER_TAG in (tags or ())
        for batch in response.generations:
            for generation in batch:
                if not isinstance(generation, ChatGeneration):
                    continue
                message = generation.message
                if not isinstance(message, AIMessage):
                    continue
                usage = message.usage_metadata or {}
                reported_cost = _extract_reported_cost(message)
                # Both OpenAI adapters may omit usage on a successful response.
                if not usage and reported_cost is None:
                    continue

                model_name = _extract_model_name(
                    message,
                    fallback=(response.llm_output or {}).get("model_name")
                    if response.llm_output
                    else None,
                )
                self.record.add_usage(
                    model_name=model_name,
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)),
                    cached_tokens=_extract_cached_tokens(message),
                    reported_cost=reported_cost,
                    driver=is_driver,
                )


def save_cost_record(datapoint_dir: Path, record: CostRecord) -> Path:
    """Write ``cost.json`` inside *datapoint_dir* and return the path."""
    datapoint_dir.mkdir(parents=True, exist_ok=True)
    cost_file = datapoint_dir / COST_FILENAME
    cost_file.write_text(
        json.dumps(record.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return cost_file


def load_cost_record(datapoint_dir: Path) -> CostRecord | None:
    """Read ``cost.json`` from *datapoint_dir*; return ``None`` if absent or corrupt."""
    cost_file = datapoint_dir / COST_FILENAME
    if not cost_file.exists() or cost_file.stat().st_size == 0:
        return None
    try:
        return CostRecord.model_validate_json(cost_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None
