"""Independent tool-schema and execution-boundary ablation controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ToolContractConfig:
    """Controls what the model sees and what the execution boundary enforces.

    These switches are deliberately independent: exposing ``minLength`` to a
    provider does not imply that the provider enforces it, and runtime
    validation checks only the schema actually supplied by the backend.
    """

    provider_schema_strict: bool = False
    runtime_schema_validation: bool = False

    @classmethod
    def from_value(cls, value: "ToolContractConfig | dict[str, Any] | None") -> "ToolContractConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("tool contract config must be a ToolContractConfig or mapping")
        unknown = set(value).difference(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown tool contract options: {sorted(unknown)}")
        if any(not isinstance(item, bool) for item in value.values()):
            raise TypeError("tool contract options must be booleans")
        return cls(**value)

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


def validate_tool_arguments(tool_definition: dict[str, Any], arguments: Any) -> list[str]:
    """Validate the small JSON Schema subset used by EvoSAGE tool contracts."""
    function = tool_definition.get("function", {})
    name = function.get("name", "<unknown>")
    schema = function.get("parameters", {})
    if not isinstance(arguments, dict):
        return ["arguments must be a JSON object"]

    errors: list[str] = []
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for key in required:
        if key not in arguments:
            errors.append(f"missing required argument: {key}")
    if schema.get("additionalProperties") is False:
        for key in arguments:
            if key not in properties:
                errors.append(f"unexpected argument: {key}")

    type_checks = {
        "string": lambda value: isinstance(value, str),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda value: isinstance(value, bool),
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
    }
    for key, value in arguments.items():
        prop = properties.get(key)
        if not isinstance(prop, dict):
            continue
        expected = prop.get("type")
        if expected in type_checks and not type_checks[expected](value):
            errors.append(f"argument {key} must be {expected}")
            continue
        min_length = prop.get("minLength")
        if isinstance(value, str) and isinstance(min_length, int) and len(value) < min_length:
            errors.append(f"argument {key} length must be at least {min_length}")

    if errors:
        return [f"{name}: {error}" for error in errors]
    return []
