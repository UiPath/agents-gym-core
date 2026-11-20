"""A model that fails to instantiate is reported without the secrets in its config."""

import pytest

from agents_gym_core.models.model_spec import ModelSpec


def test_a_failed_instantiation_does_not_print_secrets() -> None:
    def broken(**config: object) -> None:
        raise RuntimeError("boom")

    spec = ModelSpec("some-model", broken)  # type: ignore[arg-type]

    with pytest.raises(ValueError) as error:
        spec._instantiate({"model": "some-model", "api_key": "sk-secret", "max_tokens": 5})

    assert "sk-secret" not in str(error.value)
    assert "'max_tokens': 5" in str(error.value)
