"""Tests for SchemaBuilder: Pydantic schema creation from agent YAML config."""

from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from agents_gym_core.hydra_agent.schema_builder import SchemaBuilder


def _cfg(d: dict[str, Any]) -> DictConfig:
    return OmegaConf.create(d)


class TestCreateSchemaFromConfig:
    """create_schema_from_config builds a Pydantic model from field definitions."""

    @pytest.mark.parametrize(
        ("type_name", "python_type", "wrong"),
        [("str", str, 1), ("int", int, "five"), ("float", float, "pi"), ("bool", bool, "maybe")],
    )
    def test_field_type(self, type_name: str, python_type: type, wrong: object) -> None:
        schema = SchemaBuilder.create_schema_from_config(
            _cfg({"f": {"type": type_name, "description": "A field"}}), "Test"
        )
        assert schema.model_fields["f"].annotation is python_type
        with pytest.raises(ValidationError):
            schema(f=wrong)

    @pytest.mark.parametrize("default", [None, "None"])
    def test_a_none_default_makes_the_field_optional(self, default: str | None) -> None:
        schema = SchemaBuilder.create_schema_from_config(
            _cfg({"note": {"type": "str", "description": "Optional", "default": default}}),
            "Test",
        )
        assert schema.model_fields["note"].annotation == str | None
        assert schema().note is None  # type: ignore[attr-defined]

    def test_default_value(self) -> None:
        schema = SchemaBuilder.create_schema_from_config(
            _cfg({"mode": {"type": "str", "description": "Mode", "default": "fast"}}),
            "Test",
        )
        inst = schema()
        assert inst.mode == "fast"  # type: ignore[attr-defined]

    def test_simple_type_annotation(self) -> None:
        schema = SchemaBuilder.create_schema_from_config(_cfg({"value": "str"}), "Test")
        inst = schema(value="hello")
        assert inst.value == "hello"  # type: ignore[attr-defined]

    def test_invalid_config_type_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a DictConfig or dict"):
            SchemaBuilder.create_schema_from_config("not_a_dict", "Bad")  # type: ignore[arg-type]


class TestCreateSchema:
    """_create_schema handles different config formats."""

    def test_fields_wrapper(self) -> None:
        schema = SchemaBuilder._create_schema(
            _cfg({"fields": {"x": {"type": "int", "description": "X"}}}), "Test"
        )
        inst = schema(x=1)
        assert inst.x == 1  # type: ignore[attr-defined]

    def test_direct_fields(self) -> None:
        schema = SchemaBuilder._create_schema(
            _cfg({"x": {"type": "int", "description": "X"}}), "Test"
        )
        inst = schema(x=1)
        assert inst.x == 1  # type: ignore[attr-defined]


class TestCreateInputOutputSchemas:
    """create_input_output_schemas builds both schemas from agent config."""

    def test_calculator_style_config(self) -> None:
        cfg = _cfg(
            {
                "agent": {
                    "input_schema": {
                        "fields": {"expression": {"type": "str", "description": "Math expression"}},
                    },
                    "output_schema": {
                        "fields": {"answer": {"type": "int", "description": "Result"}},
                    },
                }
            }
        )
        input_schema, output_schema = SchemaBuilder.create_input_output_schemas(cfg)

        inp = input_schema(expression="2+2")
        assert inp.expression == "2+2"  # type: ignore[attr-defined]

        out = output_schema(answer=4)
        assert out.answer == 4  # type: ignore[attr-defined]

        with pytest.raises(ValidationError):
            output_schema()
