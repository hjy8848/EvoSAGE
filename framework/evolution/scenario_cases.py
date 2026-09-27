"""Adapters that expose scenario PathLists to split construction.

This is dataset infrastructure only: it does not claim that every scenario has
been validated for real-model co-evolution.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol


class ScenarioCaseProvider(Protocol):
    """Minimal contract required to build deterministic scenario cases."""

    def generate_paths(self) -> list[dict[str, Any]]: ...

    def intent_path_mapping(self) -> dict[str, dict[str, Any]]: ...


class PathListScenarioCaseProvider:
    """Adapt one existing ``*_PathList`` module without changing its truth."""

    def __init__(self, module_name: str):
        self.module = import_module(f"..sop.{module_name}", package=__package__)

    def generate_paths(self) -> list[dict[str, Any]]:
        return self.module.generate_path_list()

    def intent_path_mapping(self) -> dict[str, dict[str, Any]]:
        return self.module.get_intent_path_mapping()


_SCENARIO_PATHLIST_MODULES = {
    "ecommerce_refund": "ecommerce_refund_PathList",
    "telecom_package": "telecom_package_PathList",
    "property_service": "property_service_PathList",
    "logistics_delivery": "logistics_delivery_PathList",
    "airline_refund": "airline_refund_PathList",
    "online_education": "online_education_PathList",
}


def get_case_provider(scenario: str) -> ScenarioCaseProvider:
    """Return the PathList adapter registered for ``scenario``."""
    try:
        module_name = _SCENARIO_PATHLIST_MODULES[scenario]
    except KeyError as exc:
        raise ValueError(f"no scenario case provider registered for {scenario!r}") from exc
    return PathListScenarioCaseProvider(module_name)
