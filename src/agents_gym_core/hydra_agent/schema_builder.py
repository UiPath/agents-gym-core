"""Builder for Pydantic schemas from configuration."""

from typing import Any

from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, create_model

from agents_gym_core.utils.general_utils import get_python_type


class SchemaBuilder:
    """Builder for Pydantic schemas from configuration."""

    @staticmethod
    def create_schema_from_config(schema_config: DictConfig, schema_name: str) -> type[BaseModel]:
        """Create a Pydantic model named ``schema_name`` from field definitions.

        Each field maps to ``{type, description, default}`` (type strings as in
        :func:`get_python_type`) or to a bare type string. A ``None`` default makes the field
        optional; a field without a default is required. Example::

            expression:
              type: "str"
              description: "Mathematical expression to evaluate"
        """
        if not (OmegaConf.is_config(schema_config) or isinstance(schema_config, dict)):
            raise ValueError(
                f"Schema config must be a DictConfig or dict, got {type(schema_config)}: {schema_config}"
            )

        schema_dict = OmegaConf.to_container(schema_config, resolve=True)
        if not isinstance(schema_dict, dict):
            raise ValueError(f"Schema config must be a dictionary, got {type(schema_dict)}")

        fields: dict[str, Any] = {}

        for field_name_raw, field_def in schema_dict.items():
            field_name = str(field_name_raw)
            if isinstance(field_def, dict):
                field_type_str = str(field_def.get("type", "str"))
                field_type: type = get_python_type(field_type_str)

                description = str(field_def.get("description", ""))

                default = field_def.get("default", ...)
                if default == "None" or default is None:
                    field_type = field_type | None  # type: ignore[assignment]
                    default = None

                if description:
                    fields[field_name] = (
                        field_type,
                        Field(description=description, default=default),
                    )
                else:
                    if default is ...:
                        fields[field_name] = (field_type, ...)
                    else:
                        fields[field_name] = (field_type, Field(default=default))
            else:
                field_type = get_python_type(str(field_def))
                fields[field_name] = (field_type, ...)

        return create_model(schema_name, **fields)

    @classmethod
    def _create_schema(cls, schema_config: DictConfig, schema_name: str) -> type[BaseModel]:
        """Return the class named by ``_target_``, or a model built from the field definitions.

        The field definitions may be wrapped as ``{fields: {...}}``.
        """
        schema_dict = OmegaConf.to_container(schema_config, resolve=True)

        if isinstance(schema_dict, dict) and "_target_" in schema_dict:
            return get_class(schema_dict["_target_"])

        if isinstance(schema_dict, dict) and "fields" in schema_dict:
            fields_config = schema_config.fields
            if isinstance(fields_config, str):
                raise ValueError(
                    f"{schema_name.lower()}.fields must be a dict/DictConfig, got string: {fields_config}"
                )
            return cls.create_schema_from_config(fields_config, schema_name)

        return cls.create_schema_from_config(schema_config, schema_name)

    @classmethod
    def create_input_output_schemas(
        cls, cfg: DictConfig
    ) -> tuple[type[BaseModel], type[BaseModel]]:
        """Return ``(input_schema, output_schema)`` built from ``cfg.agent``."""
        input_schema = cls._create_schema(cfg.agent.input_schema, "InputSchema")
        output_schema = cls._create_schema(cfg.agent.output_schema, "OutputSchema")
        return input_schema, output_schema
