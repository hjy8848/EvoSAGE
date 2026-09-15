"""Case and backend factories used by the evaluation pipeline."""

from typing import Any, Dict, Optional
import hashlib

from .base import BackendEnvironment
from .ecommerce import EcommerceBackend
from .scenario import ScenarioBackend
from .types import CaseSpec
from ..config import get_scenario_config


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}-{digest}"


def _build_legacy_gt(scenario_id: str, path_config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize PathList truth into the evaluator's stable legacy shape."""
    values = path_config.get("Classification_items", [])
    field_names = list(get_scenario_config(scenario_id).classification_fields.keys())
    classification = {
        field_name: values[index] if index < len(values) else None
        for index, field_name in enumerate(field_names)
    }
    return {
        "classification": classification,
        "expected_path": list(path_config.get("expected_path", [])),
        "finals": dict(path_config.get("final_output", {})),
    }


def build_case_spec(
    scenario_id: str,
    user_intent: str,
    path_config: Optional[Dict[str, Any]] = None,
    user_id: str = "user",
    user_policy_mode: str = "truthful",
) -> CaseSpec:
    """Build a deterministic case from the existing PathList.

    Ecommerce is fully backed by EcommerceBackend in this migration. Other
    scenarios receive a compatible CaseSpec so the pipeline can migrate them
    incrementally without changing the public runner API.
    """
    path_config = path_config or {}
    legacy_gt = _build_legacy_gt(scenario_id, path_config)
    case_id = _stable_id("CASE", f"{scenario_id}:{user_intent}:{user_id}")
    expected_action = path_config.get("final_output", {}).get("Action", "")

    if scenario_id == "ecommerce_refund":
        system_variables = path_config.get("system_variables", {})
        classification = path_config.get("Classification_items", [])
        responsibility = classification[2] if len(classification) > 2 else "User"
        reason = classification[3] if len(classification) > 3 else "Reasonable"
        has_document = classification[1] if len(classification) > 1 else None
        order_id = _stable_id("ORD", case_id)
        customer_id = _stable_id("CUS", case_id)
        backend_record = {
            "order": {
                "order_id": order_id,
                "product": {"product_id": "SKU-001", "name": "无线耳机", "price": 299},
                "shipping_status": system_variables.get("ShippingStatus", "Signed"),
                "payment_status": "Paid",
                "paid_amount": 299,
                "refund_status": "None",
                "refund_eligible": expected_action in {"Refund", "Interception", "CollectionService"},
                "return_window_open": True,
                "responsibility": responsibility,
                "refund_reasonable": reason,
                "has_document": has_document,
            },
            "customer": {
                "customer_id": customer_id,
                "credit_level": system_variables.get("CreditLevel", "Medium"),
            },
        }
        expected_outcome = {"order.last_action": expected_action} if expected_action else {}
        if expected_action == "Refund":
            expected_outcome["order.refund_status"] = "Approved"
        elif expected_action == "Interception":
            expected_outcome["order.interception_status"] = "Requested"
        elif expected_action == "CollectionService":
            expected_outcome["order.return_status"] = "PickupScheduled"
        elif expected_action == "Reject":
            expected_outcome["order.refund_status"] = "Rejected"

        actual_shipping_status = system_variables.get("ShippingStatus", "Signed")
        if user_policy_mode in {"mistaken", "adversarial_false_claim"}:
            believed_shipping_status = (
                "Unshipped" if actual_shipping_status != "Unshipped" else "Signed"
            )
        else:
            believed_shipping_status = actual_shipping_status
        reveal_order_id = user_policy_mode != "withholding"

        return CaseSpec(
            case_id=case_id,
            scenario=scenario_id,
            backend_record=backend_record,
            user_goal={
                "type": "refund" if "refund" in user_intent or "refund" in expected_action.lower() else user_intent,
                "desired_action": expected_action,
            },
            user_knowledge={
                "order_id": order_id,
                "customer_id": customer_id,
                "product_name": "无线耳机",
                "knows_order_id": True,
                "knows_customer_id": True,
                "believes_shipping_status": believed_shipping_status,
            },
            user_policy={
                "mode": user_policy_mode,
                "truthfulness": "truthful" if user_policy_mode == "truthful" else "unreliable",
                "reveal_order_id_on_request": True,
                "reveal_customer_id_on_request": True,
                "show_order_id_initially": reveal_order_id,
                "reveal_hidden_account_fields": False,
            },
            initial_observation={},
            expected_outcome=expected_outcome,
            metadata={
                "user_intent": user_intent,
                "path_config": path_config,
                "legacy_path_config": path_config,
                "legacy_gt": legacy_gt,
                # Keep the flat aliases for existing consumers.
                "classification_dict": legacy_gt["classification"],
                "expected_path": legacy_gt["expected_path"],
                "finals": legacy_gt["finals"],
                "classification": classification,
            },
        )

    # All remaining scenarios use the same deterministic record contract; the
    # adapter exposes only scenario-specific tools and public fields.
    record_id = _stable_id("REC", f"{case_id}:record")
    customer_id = _stable_id("CUS", case_id)
    expected_outcome = {"interaction.last_action": expected_action} if expected_action else {}
    if expected_action == "PLAN":
        expected_outcome["interaction.plan_created"] = True
    return CaseSpec(
        case_id=case_id,
        scenario=scenario_id,
        backend_record={
            "record": {"record_id": record_id, "customer_id": customer_id},
            "system_info": dict(path_config.get("system_variables", {})),
            "status": {},
            "interaction": {
                "last_action": None,
                "status": "open",
                "answer_completed": False,
            },
        },
        user_goal={"type": user_intent, "desired_action": expected_action},
        user_knowledge={
            "record_id": record_id,
            "customer_id": customer_id,
            "knows_record_id": True,
            "knows_customer_id": True,
        },
        user_policy={"truthfulness": "truthful", "reveal_record_id_on_request": True},
        initial_observation={},
        expected_outcome=expected_outcome,
        metadata={
            "user_intent": user_intent,
            "path_config": path_config,
            "legacy_path_config": path_config,
            "legacy_gt": legacy_gt,
            "classification_dict": legacy_gt["classification"],
            "expected_path": legacy_gt["expected_path"],
            "finals": legacy_gt["finals"],
        },
    )


def create_backend(case_spec: CaseSpec) -> BackendEnvironment:
    if case_spec.scenario == "ecommerce_refund":
        return EcommerceBackend(case_spec)
    return ScenarioBackend(case_spec)
