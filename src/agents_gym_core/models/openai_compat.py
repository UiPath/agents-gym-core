"""Direct OpenAI-compatible clients for the ``langchain`` backend.

Picks Azure OpenAI when ``AZURE_OPENAI_API_KEY`` and ``AZURE_OPENAI_ENDPOINT`` are set
(``AZURE_OPENAI_API_VERSION`` overrides the spec's ``api_version``). Otherwise uses the
OpenAI API with ``OPENAI_API_KEY``; ``OPENAI_BASE_URL`` sends every OpenAI client to a compatible
server instead. To serve one self-hosted model (vLLM, a Fireworks deployment) next to OpenAI's
judges and embeddings, set ``AGENTS_GYM_MODEL_BASE_URL`` (and ``AGENTS_GYM_MODEL_API_KEY``): model
names no registry knows go there, and registered ones keep their own clients.

These classes are ``ModelSpec`` factories (``factory_method="create"``), so the registries
keep one spec per model and the choice is made from the environment at instantiation.
"""

import os
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

_AZURE_ONLY_KEYS = ("api_version", "azure_deployment", "azure_endpoint")
MODEL_BASE_URL_ENV = "AGENTS_GYM_MODEL_BASE_URL"
MODEL_API_KEY_ENV = "AGENTS_GYM_MODEL_API_KEY"


def use_azure_openai() -> bool:
    """True when the environment selects Azure OpenAI over the OpenAI API."""
    return bool(os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"))


def _azure_config(config: dict[str, Any]) -> dict[str, Any]:
    if api_version := os.environ.get("AZURE_OPENAI_API_VERSION"):
        return {**config, "api_version": api_version}
    return config


def _openai_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in _AZURE_ONLY_KEYS}


class OpenAIOrAzureChat:
    """Chat model factory: ``AzureChatOpenAI`` or ``ChatOpenAI``, chosen from the environment."""

    @classmethod
    def create(cls, **config: Any) -> BaseChatModel:
        if use_azure_openai():
            return AzureChatOpenAI(**_azure_config(config))
        return ChatOpenAI(**_openai_config(config))


class SelfHostedChat:
    """Chat model factory: ``ChatOpenAI`` on ``AGENTS_GYM_MODEL_BASE_URL``, for unregistered names.

    The key comes from ``AGENTS_GYM_MODEL_API_KEY`` (vLLM accepts any) when the client is created.
    """

    @classmethod
    def create(cls, **config: Any) -> BaseChatModel:
        api_key = os.environ.get(MODEL_API_KEY_ENV) or "EMPTY"
        return ChatOpenAI(
            base_url=os.environ[MODEL_BASE_URL_ENV],
            api_key=SecretStr(api_key),
            **_openai_config(config),
        )


class _KeylessEmbeddings(Embeddings):
    """Stands in when no OpenAI or Azure key is set, so runs that never embed still start."""

    def __init__(self, model: str) -> None:
        self.model = model

    def _fail(self) -> list:
        raise ValueError(
            f"embedding model {self.model} needs OPENAI_API_KEY, or AZURE_OPENAI_API_KEY and "
            "AZURE_OPENAI_ENDPOINT"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._fail()

    def embed_query(self, text: str) -> list[float]:
        return self._fail()


class OpenAIOrAzureEmbeddings:
    """Embeddings factory: ``AzureOpenAIEmbeddings`` or ``OpenAIEmbeddings``, by environment."""

    @classmethod
    def create(cls, **config: Any) -> Embeddings:
        if use_azure_openai():
            return AzureOpenAIEmbeddings(**_azure_config(config))
        if not (config.get("api_key") or os.environ.get("OPENAI_API_KEY")):
            return _KeylessEmbeddings(config.get("model", ""))
        return OpenAIEmbeddings(**_openai_config(config))
