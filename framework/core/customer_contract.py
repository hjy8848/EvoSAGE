"""Shared machine-readable disclosure contracts for Customer simulation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..backend.types import CaseSpec


def get_customer_opening_contract(
    case_spec: Optional[CaseSpec],
) -> List[Dict[str, Any]]:
    """Return disclosures that are both known to the Customer and required
    on the opening turn.

    This intentionally preserves the existing CaseSpec semantics: an order
    identifier is mandatory only when ``show_order_id_initially`` is exactly
    true or ``mandatory_opening_disclosures`` contains ``order_id``, and the
    Customer is explicitly marked as knowing a non-empty identifier.
    """
    if case_spec is None:
        return []

    knowledge = case_spec.user_knowledge or {}
    policy = case_spec.user_policy or {}
    knows_order_id = bool(knowledge.get("knows_order_id") and knowledge.get("order_id"))

    mandatory = policy.get("mandatory_opening_disclosures", [])
    if isinstance(mandatory, str):
        mandatory = [mandatory]
    order_id_required = (
        policy.get("show_order_id_initially") is True
        or "order_id" in mandatory
    )

    if not (knows_order_id and order_id_required):
        return []

    return [{
        "field": "order_id",
        "value": str(knowledge["order_id"]),
        "timing": "opening_turn",
    }]
