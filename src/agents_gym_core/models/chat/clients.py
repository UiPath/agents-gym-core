import os
import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from uipath_langchain_client.clients.anthropic.chat_models import UiPathChatAnthropic
from uipath_langchain_client.clients.bedrock.chat_models import UiPathChatAnthropicBedrock
from uipath_langchain_client.clients.google.chat_models import (
    UiPathChatGoogleGenerativeAI,
)
from uipath_langchain_client.clients.openai.chat_models import (
    UiPathAzureChatOpenAI,
)
from uipath_langchain_client.factory import get_chat_model as get_chat_model_factory

from agents_gym_core.models.model_spec import ModelRegistryMeta, ModelSpec
from agents_gym_core.models.openai_compat import MODEL_BASE_URL_ENV
from agents_gym_core.models.utils import (
    GLOBAL_DEFAULT_JUDGE_CONFIG,
    GLOBAL_DEFAULT_MODEL_CONFIG,
    UIPATH_RETRY_CONFIG,
    ModelBackend,
    default_backend,
    get_or_create_rate_limiter,
)
from agents_gym_core.utils.general_utils import hashable_lru_cache
from agents_gym_core.utils.gym_settings_core import gym_settings_core
from agents_gym_core.utils.trace_utils import warning

# Claude Opus 4.6 gets the same output cap on every backend: without one the gateway's Bedrock
# client falls back to 4096 tokens and the direct client to the model's 128000.
OPUS_4_6_CONFIG: dict[str, Any] = {
    "max_tokens": gym_settings_core.UIPATH_AGENTS_GYM_MAX_TOKENS or 64_000
}


def _direct_anthropic_name(model_name: str) -> str:
    """Map a Bedrock or Vertex Claude name to the Anthropic API's, for the langchain backend.

    ``anthropic.claude-opus-4-6-v1`` (Bedrock) and ``claude-sonnet-4-5@20250929`` (Vertex) name
    models that the gateway serves; with provider keys they are reached as ``claude-opus-4-6`` /
    ``claude-sonnet-4-5-20250929``. Unknown names pass through.
    """
    name = re.sub(r"-v\d+(:\d+)?$", "", model_name.removeprefix("anthropic.")).replace("@", "-")
    return name if name != model_name and name in LangchainAnthropicModels else model_name


@hashable_lru_cache
def get_chat_model(
    model_name: str,
    max_concurrency: int | None = None,
    is_judge: bool = False,
    from_factory: bool = False,
    backend: ModelBackend | str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Get chat model instance by name.

    ``backend`` defaults to :func:`~agents_gym_core.models.utils.default_backend`.
    """
    backend = ModelBackend(backend) if backend else default_backend()
    _local_registries = (HuggingFaceChatModels, VllmChatModels, LangchainFireworksModels)
    registries = (
        (LangchainOpenAIModels, LangchainAnthropicModels, LangchainGeminiModels)
        if backend == "langchain"
        else (
            OpenAIModels,
            GeminiModels,
            AnthropicBedrockModels,
            AnthropicVertexModels,
        )
    ) + _local_registries
    if backend == "langchain":
        model_name = _direct_anthropic_name(model_name)
    if not from_factory:
        for registry in registries:
            if model_name in registry:
                return registry[model_name].get_chat_model(is_judge=is_judge, **kwargs)

    if max_concurrency is None:
        max_concurrency = gym_settings_core.UIPATH_AGENTS_GYM_MAX_LLM_CONCURRENCY

    if backend == ModelBackend.LANGCHAIN:
        # An unregistered name (e.g. a model served by vLLM) goes to AGENTS_GYM_MODEL_BASE_URL
        # when set, else to the direct OpenAI-compatible client; never to the UiPath gateway.
        factory = "SelfHostedChat" if os.environ.get(MODEL_BASE_URL_ENV) else "OpenAIOrAzureChat"
        return ModelSpec(
            model_name,
            f"agents_gym_core.models.openai_compat:{factory}",
            factory_method="create",
            max_concurrency=max_concurrency,
        ).get_chat_model(is_judge=is_judge, **kwargs)

    rate_limiter = None
    if max_concurrency is not None:
        rate_limiter = get_or_create_rate_limiter(model_name, max_concurrency)

    try:
        if not from_factory:
            warning(
                f"Model {model_name} not found in any registry, attempting to get it directly from the client factory."
            )
        config: dict[str, Any] = {
            **(GLOBAL_DEFAULT_JUDGE_CONFIG if is_judge else GLOBAL_DEFAULT_MODEL_CONFIG),
            **kwargs,
        }
        if backend == ModelBackend.UIPATH:
            config.setdefault("retry_config", UIPATH_RETRY_CONFIG)
        model = get_chat_model_factory(
            model_name,
            rate_limiter=rate_limiter,
            **config,
        )
    except ValueError as e:
        raise ValueError(f"Model {model_name} not found in any registry") from e

    return model


class OpenAIModels(metaclass=ModelRegistryMeta):
    """Azure OpenAI models."""

    _base_config: dict[str, Any] = {
        "api_version": "2025-03-01-preview",
        "retry_config": UIPATH_RETRY_CONFIG,
    }
    _reasoning_config: dict[str, Any] = {
        **_base_config,
        "use_responses_api": True,
        "reasoning": {"effort": "low", "summary": "auto"},
        "verbosity": "medium",
    }

    gpt_4o_2024_11_20 = ModelSpec(
        "gpt-4o-2024-11-20", UiPathAzureChatOpenAI, default_config=_base_config
    )

    gpt_4_1_2025_04_14 = ModelSpec(
        "gpt-4.1-2025-04-14", UiPathAzureChatOpenAI, default_config=_base_config
    )
    gpt_4_1_mini_2025_04_14 = ModelSpec(
        "gpt-4.1-mini-2025-04-14", UiPathAzureChatOpenAI, default_config=_base_config
    )
    gpt_4_1_nano_2025_04_14 = ModelSpec(
        "gpt-4.1-nano-2025-04-14", UiPathAzureChatOpenAI, default_config=_base_config
    )

    gpt_5_2025_08_07 = ModelSpec(
        "gpt-5-2025-08-07",
        UiPathAzureChatOpenAI,
        default_config=_reasoning_config,
    )
    gpt_5_mini_2025_08_07 = ModelSpec(
        "gpt-5-mini-2025-08-07",
        UiPathAzureChatOpenAI,
        default_config=_reasoning_config,
    )
    gpt_5_nano_2025_08_07 = ModelSpec(
        "gpt-5-nano-2025-08-07",
        UiPathAzureChatOpenAI,
        default_config=_reasoning_config,
    )

    gpt_5_1_2025_11_13 = ModelSpec(
        "gpt-5.1-2025-11-13",
        UiPathAzureChatOpenAI,
        default_config=_reasoning_config,
    )

    gpt_5_2_2025_12_11 = ModelSpec(
        "gpt-5.2-2025-12-11",
        UiPathAzureChatOpenAI,
        default_config=_reasoning_config,
    )

    gpt_5_4_2026_03_05 = ModelSpec(
        "gpt-5.4-2026-03-05",
        UiPathAzureChatOpenAI,
        default_config=_reasoning_config,
    )
    gpt_5_4_mini_2026_03_17 = ModelSpec(
        "gpt-5.4-mini-2026-03-17",
        UiPathAzureChatOpenAI,
        default_config=_reasoning_config,
    )
    gpt_5_4_nano_2026_03_17 = ModelSpec(
        "gpt-5.4-nano-2026-03-17",
        UiPathAzureChatOpenAI,
        default_config=_reasoning_config,
    )
    gpt_5_4_pro_2026_03_05 = ModelSpec(
        "gpt-5.4-pro-2026-03-05",
        UiPathAzureChatOpenAI,
        default_config=_reasoning_config,
    )

    gpt_5_5_2026_04_24 = ModelSpec(
        "gpt-5.5-2026-04-24",
        UiPathAzureChatOpenAI,
        default_config=_reasoning_config,
    )

    gpt_5_6_luna_2026_07_09 = ModelSpec(
        "gpt-5.6-luna-2026-07-09",
        UiPathAzureChatOpenAI,
        default_config=_reasoning_config,
    )
    gpt_5_6_sol_2026_07_09 = ModelSpec(
        "gpt-5.6-sol-2026-07-09",
        UiPathAzureChatOpenAI,
        default_config=_reasoning_config,
    )
    gpt_5_6_terra_2026_07_09 = ModelSpec(
        "gpt-5.6-terra-2026-07-09",
        UiPathAzureChatOpenAI,
        default_config=_reasoning_config,
    )


class GeminiModels(metaclass=ModelRegistryMeta):
    """Google Vertex AI models."""

    _default_config: dict[str, Any] = {"retry_config": UIPATH_RETRY_CONFIG}
    _kwarg_map: dict[str, str | None] = {"max_tokens": "max_output_tokens"}

    gemini_2_5_pro = ModelSpec(
        "gemini-2.5-pro",
        UiPathChatGoogleGenerativeAI,
        default_config=_default_config,
        max_concurrency=5,
        kwarg_map=_kwarg_map,
    )
    gemini_2_5_flash = ModelSpec(
        "gemini-2.5-flash",
        UiPathChatGoogleGenerativeAI,
        default_config=_default_config,
        max_concurrency=5,
        kwarg_map=_kwarg_map,
    )

    gemini_3_flash_preview = ModelSpec(
        "gemini-3-flash-preview",
        UiPathChatGoogleGenerativeAI,
        default_config=_default_config,
        max_concurrency=5,
        kwarg_map=_kwarg_map,
    )
    gemini_3_pro_preview = ModelSpec(
        "gemini-3-pro-preview",
        UiPathChatGoogleGenerativeAI,
        default_config=_default_config,
        max_concurrency=5,
        kwarg_map=_kwarg_map,
    )


class AnthropicBedrockModels(metaclass=ModelRegistryMeta):
    """Anthropic Bedrock models."""

    _default_config: dict[str, Any] = {
        "vendor_type": "awsbedrock",
        "retry_config": UIPATH_RETRY_CONFIG,
    }
    _kwarg_map: dict[str, str | None] = {"request_timeout": "timeout"}

    anthropic_claude_sonnet_4_5 = ModelSpec(
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        UiPathChatAnthropicBedrock,
        max_concurrency=5,
        kwarg_map=_kwarg_map,
    )
    anthropic_claude_haiku_4_5 = ModelSpec(
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        UiPathChatAnthropicBedrock,
        max_concurrency=10,
        kwarg_map=_kwarg_map,
    )
    anthropic_claude_opus_4_5 = ModelSpec(
        "anthropic.claude-opus-4-5-20251101-v1:0",
        UiPathChatAnthropicBedrock,
        max_concurrency=5,
        kwarg_map=_kwarg_map,
    )
    anthropic_claude_sonnet_4_6 = ModelSpec(
        "anthropic.claude-sonnet-4-6",
        UiPathChatAnthropicBedrock,
        max_concurrency=5,
        kwarg_map=_kwarg_map,
    )
    anthropic_claude_opus_4_6 = ModelSpec(
        "anthropic.claude-opus-4-6-v1",
        UiPathChatAnthropicBedrock,
        max_concurrency=5,
        default_config=OPUS_4_6_CONFIG,
        kwarg_map=_kwarg_map,
    )
    # Opus 4.7 drops the "-v1" suffix 4.6 still carries and rejects
    # ``temperature`` ("deprecated for this model"); strip it via kwarg_map.
    anthropic_claude_opus_4_7 = ModelSpec(
        "anthropic.claude-opus-4-7",
        UiPathChatAnthropicBedrock,
        max_concurrency=5,
        kwarg_map={**_kwarg_map, "temperature": None},
    )


class AnthropicVertexModels(metaclass=ModelRegistryMeta):
    """Anthropic Vertex models."""

    _default_config: dict[str, Any] = {
        "vendor_type": "vertexai",
        "retry_config": UIPATH_RETRY_CONFIG,
    }
    _kwarg_map: dict[str, str | None] = {"request_timeout": "timeout"}

    anthropic_claude_sonnet_4_5 = ModelSpec(
        "claude-sonnet-4-5@20250929",
        UiPathChatAnthropic,
        max_concurrency=5,
        default_config=_default_config,
        kwarg_map=_kwarg_map,
    )
    anthropic_claude_haiku_4_5 = ModelSpec(
        "claude-haiku-4-5@20251001",
        UiPathChatAnthropic,
        max_concurrency=10,
        default_config=_default_config,
        kwarg_map=_kwarg_map,
    )
    anthropic_claude_opus_4_5 = ModelSpec(
        "claude-opus-4-5@20251101",
        UiPathChatAnthropic,
        max_concurrency=5,
        default_config=_default_config,
        kwarg_map=_kwarg_map,
    )
    anthropic_claude_sonnet_4_6 = ModelSpec(
        "claude-sonnet-4-6",
        UiPathChatAnthropic,
        max_concurrency=5,
        default_config=_default_config,
        kwarg_map=_kwarg_map,
    )
    anthropic_claude_opus_4_6 = ModelSpec(
        "claude-opus-4-6",
        UiPathChatAnthropic,
        max_concurrency=5,
        default_config={**_default_config, **OPUS_4_6_CONFIG},
        kwarg_map=_kwarg_map,
    )


class HuggingFaceChatModels(metaclass=ModelRegistryMeta):
    """Local HuggingFace chat models, built with ``from_model_id`` on a HuggingFacePipeline.

    API-level globals (temperature, max_tokens, ...) are stripped; sampling settings live in
    ``pipeline_kwargs``. No rate limiter, since inference is local.
    """

    qwen3_5_2b = ModelSpec(
        "Qwen/Qwen3.5-2B",
        "agents_gym_core.models.chat.huggingface:PatchedChatHuggingFace",
        default_config={
            "task": "text-generation",
            "device_map": "auto",
            "model_kwargs": {"dtype": "float16", "attn_implementation": "sdpa"},
            "pipeline_kwargs": {
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature": 0.1,
                "return_full_text": False,
            },
        },
        max_concurrency=None,
        model_param="model_id",
        factory_method="from_model_id",
        kwarg_map={
            "max_tokens": None,
            "request_timeout": None,
            "max_retries": None,
            "temperature": None,
        },
    )


class VllmChatModels(metaclass=ModelRegistryMeta):
    """Open-weight models behind an OpenAI-compatible endpoint: your own vLLM server, or a
    hosted provider such as Fireworks.

    The endpoint's address, its model name and its key come from the environment, so the
    registry name identifies the model in results wherever it is served. Endpoint keys use
    ``SecretStr`` to mask their values in configuration representations.
    """

    # Qwen3.5-9B with thinking off.
    # QWEN_9B_VLLM_API_KEY is for an endpoint that needs one, e.g. Fireworks.
    qwen3_5_9b = ModelSpec(
        os.environ.get("QWEN_9B_VLLM_MODEL", "Qwen/Qwen3.5-9B"),
        "langchain_openai:ChatOpenAI",
        alias="Qwen/Qwen3.5-9B",
        default_config={
            "base_url": os.environ.get("QWEN_9B_VLLM_BASE_URL", "http://localhost:8002/v1"),
            "api_key": SecretStr(os.environ.get("QWEN_9B_VLLM_API_KEY") or "EMPTY"),
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        max_concurrency=int(os.environ.get("QWEN_9B_VLLM_MAX_CONCURRENCY", "50")),
    )

    # DeepSeek-V4.1-Flash-1017; thinking follows DEEPSEEK_V41_FLASH_VLLM_THINKING (on unless "0").
    deepseek_v4_1_flash = ModelSpec(
        os.environ.get("DEEPSEEK_V41_FLASH_VLLM_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash-1017"),
        "langchain_openai:ChatOpenAI",
        alias="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        default_config={
            "base_url": os.environ.get(
                "DEEPSEEK_V41_FLASH_VLLM_BASE_URL", "http://localhost:8003/v1"
            ),
            "api_key": SecretStr(os.environ.get("DEEPSEEK_V41_FLASH_VLLM_API_KEY") or "EMPTY"),
            "extra_body": {
                "chat_template_kwargs": {
                    "thinking": os.environ.get("DEEPSEEK_V41_FLASH_VLLM_THINKING", "1") != "0"
                }
            },
        },
        max_concurrency=int(os.environ.get("DEEPSEEK_V41_FLASH_VLLM_MAX_CONCURRENCY", "50")),
    )


class LangchainOpenAIModels(metaclass=ModelRegistryMeta):
    """OpenAI models via direct API: Azure OpenAI or the OpenAI API / an OpenAI-compatible server.

    See :mod:`agents_gym_core.models.openai_compat` for how the environment picks one.
    """

    _base_config: dict[str, Any] = {"api_version": "2025-03-01-preview"}
    _reasoning_config: dict[str, Any] = {
        **_base_config,
        "use_responses_api": True,
        "reasoning": {"effort": "low", "summary": "auto"},
        "verbosity": "medium",
    }

    gpt_4o_2024_11_20 = ModelSpec(
        "gpt-4o-2024-11-20",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_base_config,
    )

    gpt_4_1_2025_04_14 = ModelSpec(
        "gpt-4.1-2025-04-14",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_base_config,
    )
    gpt_4_1_mini_2025_04_14 = ModelSpec(
        "gpt-4.1-mini-2025-04-14",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_base_config,
    )
    gpt_4_1_nano_2025_04_14 = ModelSpec(
        "gpt-4.1-nano-2025-04-14",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_base_config,
    )

    gpt_5_2025_08_07 = ModelSpec(
        "gpt-5-2025-08-07",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )
    gpt_5_mini_2025_08_07 = ModelSpec(
        "gpt-5-mini-2025-08-07",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )
    gpt_5_nano_2025_08_07 = ModelSpec(
        "gpt-5-nano-2025-08-07",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )

    gpt_5_1_2025_11_13 = ModelSpec(
        "gpt-5.1-2025-11-13",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )
    gpt_5_2_2025_12_11 = ModelSpec(
        "gpt-5.2-2025-12-11",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )

    gpt_5_4_2026_03_05 = ModelSpec(
        "gpt-5.4-2026-03-05",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )
    gpt_5_4_mini_2026_03_17 = ModelSpec(
        "gpt-5.4-mini-2026-03-17",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )
    gpt_5_4_nano_2026_03_17 = ModelSpec(
        "gpt-5.4-nano-2026-03-17",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )
    gpt_5_4_pro_2026_03_05 = ModelSpec(
        "gpt-5.4-pro-2026-03-05",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )

    gpt_5_5_2026_04_23 = ModelSpec(
        "gpt-5.5-2026-04-23",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )

    gpt_5_6_luna = ModelSpec(
        "gpt-5.6-luna",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )
    gpt_5_6_sol = ModelSpec(
        "gpt-5.6-sol",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )
    gpt_5_6_terra = ModelSpec(
        "gpt-5.6-terra",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )

    gpt_6_astra = ModelSpec(
        "gpt-6-astra",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )
    gpt_6_sol = ModelSpec(
        "gpt-6-sol",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )
    gpt_6_1_sol = ModelSpec(
        "gpt-6.1-sol",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )
    gpt_6_luna = ModelSpec(
        "gpt-6-luna",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )

    # Dated snapshots, as deployed on Azure OpenAI; the OpenAI API serves the dateless names.
    gpt_6_astra_2026_09_03 = ModelSpec(
        "gpt-6-astra-2026-09-03",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )
    gpt_6_luna_2026_09_22 = ModelSpec(
        "gpt-6-luna-2026-09-22",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )
    gpt_6_sol_2026_09_22 = ModelSpec(
        "gpt-6-sol-2026-09-22",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )
    gpt_6_1_sol_2026_09_29 = ModelSpec(
        "gpt-6.1-sol-2026-09-29",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureChat",
        factory_method="create",
        default_config=_reasoning_config,
    )


class LangchainAnthropicModels(metaclass=ModelRegistryMeta):
    """Anthropic models via direct API (requires ANTHROPIC_API_KEY)."""

    _kwarg_map: dict[str, str | None] = {"request_timeout": "timeout"}
    _no_temperature: dict[str, str | None] = {**_kwarg_map, "temperature": None}

    claude_sonnet_4_5 = ModelSpec("claude-sonnet-4-5-20250929", ChatAnthropic, kwarg_map=_kwarg_map)
    claude_haiku_4_5 = ModelSpec("claude-haiku-4-5-20251001", ChatAnthropic, kwarg_map=_kwarg_map)
    claude_opus_4_5 = ModelSpec(
        "claude-opus-4-5-20251101", ChatAnthropic, max_concurrency=5, kwarg_map=_kwarg_map
    )
    claude_sonnet_4_6 = ModelSpec(
        "claude-sonnet-4-6", ChatAnthropic, max_concurrency=5, kwarg_map=_kwarg_map
    )
    claude_opus_4_6 = ModelSpec(
        "claude-opus-4-6",
        ChatAnthropic,
        max_concurrency=5,
        default_config=OPUS_4_6_CONFIG,
        kwarg_map=_kwarg_map,
    )
    # Opus 4.7 and later reject a non-default temperature, top_p or top_k.
    claude_opus_4_7 = ModelSpec(
        "claude-opus-4-7", ChatAnthropic, max_concurrency=5, kwarg_map=_no_temperature
    )
    claude_opus_4_8 = ModelSpec(
        "claude-opus-4-8", ChatAnthropic, max_concurrency=5, kwarg_map=_no_temperature
    )
    claude_sonnet_5 = ModelSpec(
        "claude-sonnet-5", ChatAnthropic, max_concurrency=5, kwarg_map=_no_temperature
    )
    claude_opus_5 = ModelSpec(
        "claude-opus-5", ChatAnthropic, max_concurrency=5, kwarg_map=_no_temperature
    )
    claude_fable_5 = ModelSpec(
        "claude-fable-5", ChatAnthropic, max_concurrency=5, kwarg_map=_no_temperature
    )
    claude_haiku_5_5 = ModelSpec(
        "claude-haiku-5-5", ChatAnthropic, max_concurrency=5, kwarg_map=_no_temperature
    )
    claude_sonnet_5_5 = ModelSpec(
        "claude-sonnet-5-5", ChatAnthropic, max_concurrency=5, kwarg_map=_no_temperature
    )
    claude_opus_5_5 = ModelSpec(
        "claude-opus-5-5", ChatAnthropic, max_concurrency=5, kwarg_map=_no_temperature
    )
    claude_fable_5_1 = ModelSpec(
        "claude-fable-5-1", ChatAnthropic, max_concurrency=5, kwarg_map=_no_temperature
    )


class LangchainFireworksModels(metaclass=ModelRegistryMeta):
    """Fireworks AI models via direct API (requires FIREWORKS_API_KEY).

    langchain-fireworks comes with the base install (uipath-langchain-client's fireworks extra).
    """

    kimi_k2p5 = ModelSpec(
        "accounts/fireworks/models/kimi-k2p5",
        "agents_gym_core.models.chat.fireworks:PatchedChatFireworks",
        max_concurrency=10,
    )


class LangchainGeminiModels(metaclass=ModelRegistryMeta):
    """Google Gemini models via direct API (requires GOOGLE_API_KEY)."""

    _kwarg_map: dict[str, str | None] = {"max_tokens": "max_output_tokens"}

    gemini_2_5_pro = ModelSpec(
        "gemini-2.5-pro",
        ChatGoogleGenerativeAI,
        max_concurrency=5,
        kwarg_map=_kwarg_map,
    )
    gemini_2_5_flash = ModelSpec(
        "gemini-2.5-flash",
        ChatGoogleGenerativeAI,
        max_concurrency=5,
        kwarg_map=_kwarg_map,
    )
    gemini_3_1_pro_preview = ModelSpec(
        "gemini-3.1-pro-preview",
        ChatGoogleGenerativeAI,
        max_concurrency=5,
        kwarg_map=_kwarg_map,
    )
    gemini_3_8_flash = ModelSpec(
        "gemini-3.8-flash",
        ChatGoogleGenerativeAI,
        max_concurrency=5,
        kwarg_map=_kwarg_map,
    )
    gemini_4_argon = ModelSpec(
        "gemini-4-argon",
        ChatGoogleGenerativeAI,
        max_concurrency=5,
        kwarg_map=_kwarg_map,
    )
