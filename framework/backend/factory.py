"""Case and backend factories used by the evaluation pipeline."""

from typing import Any, Dict, Optional
import hashlib

from .base import BackendEnvironment
from .ecommerce import EcommerceBackend
from .scenario import SCENARIO_ACTION_STATE_UPDATES, ScenarioBackend
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


def _build_required_backend_verifications(
    scenario_id: str, path_config: Dict[str, Any]
) -> list:
    """Declare authoritative checks required by the selected path.

    These declarations drive execution scoring only.  They are stored in the
    CaseSpec for the evaluator and are never copied into the Agent prompt.
    """
    system_variables = dict(path_config.get("system_variables") or {})
    # Education PathList stores the risk branch as a top-level path field.
    # Do not require a risk query for paths whose branch does not use it.
    if scenario_id == "online_education" and isinstance(path_config.get("isRiskUser"), bool):
        system_variables["isRiskUser"] = path_config["isRiskUser"]
    requirements = []
    mappings = {
        "ecommerce_refund": {
            "ShippingStatus": ("query_order", "order_id", "shipping_status", "order_id"),
            "CreditLevel": ("query_customer_profile", "customer_id", "credit_level", "customer_id"),
            "PaymentStatus": ("query_payment", "order_id", "payment_status", "order_id"),
        },
        "telecom_package": {
            "PackageStatus": ("query_package", "record_id", "package_status", "record_id"),
            "Penalty": ("query_penalty", "record_id", "penalty", "record_id"),
        },
        "property_service": {
            "HouseStatus": ("query_property_account", "record_id", "house_status", "record_id"),
            "FeePaymentStatus": ("query_fee_status", "record_id", "fee_payment_status", "record_id"),
        },
        "logistics_delivery": {
            "orderStatus": ("query_delivery_order", "record_id", "order_status", "record_id"),
            "hasInsurance": ("query_insurance", "record_id", "has_insurance", "record_id"),
        },
        "airline_refund": {
            "memberLevel": ("query_member_profile", "record_id", "member_level", "record_id"),
            "hasInsurance": ("query_ticket_rules", "record_id", "has_insurance", "record_id"),
        },
        "online_education": {
            "isRiskUser": ("query_user_risk", "record_id", "is_risk_user", "record_id"),
        },
    }
    for backend_field, (tool, argument, result_field, knowledge_key) in mappings.get(scenario_id, {}).items():
        if backend_field not in system_variables:
            continue
        requirements.append({
            "backend_field": backend_field,
            "tool": tool,
            "argument": argument,
            "result_field": result_field,
            "knowledge_key": knowledge_key,
        })

    # Some actions have an authoritative precondition even when the selected
    # PathList does not branch on that field.  Declare those checks as well so
    # the evaluator can distinguish "the action happened" from "the action
    # was authorized by the backend state".
    expected_action = path_config.get("final_output", {}).get("Action", "")
    action_requirements = {
        "telecom_package": {
            "ChangeOrder": ("AccountStatus", "query_account", "account_status"),
        },
        "property_service": {
            "Payment": ("FeePaymentStatus", "query_fee_status", "fee_payment_status"),
            "Reject": ("FeePaymentStatus", "query_fee_status", "fee_payment_status"),
            "Registration": ("FeePaymentStatus", "query_fee_status", "fee_payment_status"),
        },
        "logistics_delivery": {
            "Interception": ("orderStatus", "query_delivery_order", "order_status"),
            "Modify": ("orderStatus", "query_delivery_order", "order_status"),
            "Registration": ("orderStatus", "query_delivery_order", "order_status"),
            "MakeUpDifference": ("orderStatus", "query_delivery_order", "order_status"),
            "Compensation": ("hasInsurance", "query_insurance", "has_insurance"),
            "TransHuman": ("orderStatus", "query_delivery_order", "order_status"),
            "Reject": ("orderStatus", "query_delivery_order", "order_status"),
            "Comfort": ("orderStatus", "query_delivery_order", "order_status"),
        },
        "airline_refund": {
            "RescheduleOrRefund": ("BookingStatus", "query_booking", "booking_status"),
            "RescheduleOrRefund+HandlingFee": ("BookingStatus", "query_booking", "booking_status"),
            "RescheduleOrRefund+Compensation": ("BookingStatus", "query_booking", "booking_status"),
            "Compensation": ("memberLevel", "query_member_profile", "member_level"),
            "TransHuman": ("memberLevel", "query_member_profile", "member_level"),
            "Reject": ("memberLevel", "query_member_profile", "member_level"),
        },
        "online_education": {
            "REVIEW": ("HistoricalComplaintRecords", "query_learning_history", "historical_complaints"),
            "NEGOTIATE": ("isRiskUser", "query_user_risk", "is_risk_user"),
            "REFUND": ("RefundEligibility", "query_refund_eligibility", "refund_eligibility"),
            "PLAN": ("KnowledgeResources", "search_course_content", "knowledge_resources"),
        },
    }
    action_requirement = action_requirements.get(scenario_id, {}).get(expected_action)
    if action_requirement:
        backend_field, tool, result_field = action_requirement
        if not any(item["tool"] == tool and item["result_field"] == result_field for item in requirements):
            requirements.append({
                "backend_field": backend_field,
                "tool": tool,
                "argument": "record_id",
                "result_field": result_field,
                "knowledge_key": "record_id",
            })

    # Paths without a branching system variable still require a record lookup;
    # otherwise an Agent could receive credit for an action on an unverified ID.
    if not requirements:
        default_tools = {
            "ecommerce_refund": ("query_order", "order_id", "order_id", "order_id"),
            "telecom_package": ("query_package", "record_id", "record_id", "record_id"),
            "property_service": ("query_property_account", "record_id", "record_id", "record_id"),
            "logistics_delivery": ("query_delivery_order", "record_id", "record_id", "record_id"),
            "airline_refund": ("query_booking", "record_id", "record_id", "record_id"),
            "online_education": ("query_course_enrollment", "record_id", "record_id", "record_id"),
        }
        if scenario_id in default_tools:
            tool, argument, result_field, knowledge_key = default_tools[scenario_id]
            requirements.append({
                "backend_field": "RecordIdentity",
                "tool": tool,
                "argument": argument,
                "result_field": result_field,
                "knowledge_key": knowledge_key,
            })
    return requirements


def _effective_system_variables(scenario_id: str, path_config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize PathList truth into backend-owned, queryable fields."""
    values = dict(path_config.get("system_variables") or {})
    if scenario_id == "online_education" and "isRiskUser" in path_config:
        values["isRiskUser"] = path_config["isRiskUser"] is True
    if scenario_id == "online_education":
        classification = path_config.get("Classification_items") or []
        # Education PathList's fifth classification field is the repeated-
        # complaint branch.  Promote it into backend state so REVIEW can be
        # verified through query_learning_history instead of inferred from
        # the model's classification claim.
        if len(classification) > 4 and isinstance(classification[4], bool):
            values["HistoricalComplaintRecords"] = classification[4]
    defaults = {
        "telecom_package": {
            "PackageStatus": "NoContract", "Penalty": 0,
            "AccountStatus": "Active", "CurrentPlan": "Standard",
        },
        "property_service": {
            "HouseStatus": "Occupied", "FeePaymentStatus": "Settled",
            "RepairTicketStatus": "None", "EmergencyLevel": "Normal",
        },
        "logistics_delivery": {
            "orderStatus": "Undelivered", "hasInsurance": False,
            "deliveryAddress": "已登记", "packageRisk": "Normal",
        },
        "airline_refund": {
            "BookingStatus": "Confirmed", "memberLevel": "Regular",
            "hasInsurance": False, "flightStatus": "Scheduled", "ticketType": "Standard",
        },
        "online_education": {
            "CourseStatus": "Enrolled", "HistoricalComplaintRecords": False,
            "RefundEligibility": True, "KnowledgeResources": ["课程资料"],
            "isRiskUser": False,
        },
    }
    for key, value in defaults.get(scenario_id, {}).items():
        values.setdefault(key, value)
    return values


def build_case_spec(
    scenario_id: str,
    user_intent: str,
    path_config: Optional[Dict[str, Any]] = None,
    user_id: str = "user",
    user_policy_mode: str = "truthful",
) -> CaseSpec:
    """Build a deterministic case from the existing PathList.

    Every supported scenario receives a deterministic CaseSpec with a hidden
    backend record, a public projection, required verification declarations,
    and an expected backend outcome.
    """
    path_config = path_config or {}
    legacy_gt = _build_legacy_gt(scenario_id, path_config)
    required_backend_verifications = _build_required_backend_verifications(
        scenario_id, path_config
    )
    case_id = _stable_id("CASE", f"{scenario_id}:{user_intent}:{user_id}")
    expected_action = path_config.get("final_output", {}).get("Action", "")

    if scenario_id == "ecommerce_refund":
        system_variables = path_config.get("system_variables", {})
        classification = path_config.get("Classification_items", [])
        shipping_status = system_variables.get("ShippingStatus", "Signed")
        payment_status = "Paid"
        responsibility = classification[2] if len(classification) > 2 else "User"
        reason = classification[3] if len(classification) > 3 else "Reasonable"
        has_document = classification[1] if len(classification) > 1 else None
        order_id = _stable_id("ORD", case_id)
        customer_id = _stable_id("CUS", case_id)
        backend_record = {
            "order": {
                "order_id": order_id,
                "product": {"product_id": "SKU-001", "name": "无线耳机", "price": 299},
                "shipping_status": shipping_status,
                "payment_status": payment_status,
                "paid_amount": 299,
                "refund_status": "None",
                # This is a backend policy fact, not a projection of the GT
                # action. It is kept internal for action validation.
                "refund_eligible": (
                    shipping_status == "Unshipped"
                    and payment_status == "Paid"
                ),
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

        actual_shipping_status = shipping_status
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
                "required_backend_verifications": required_backend_verifications,
                "backend_system_variables": dict(system_variables),
            },
        )

    # All remaining scenarios use the same deterministic record contract; the
    # adapter exposes only scenario-specific tools and public fields.
    record_id = _stable_id("REC", f"{case_id}:record")
    customer_id = _stable_id("CUS", case_id)
    system_variables = _effective_system_variables(scenario_id, path_config)
    expected_outcome = {"interaction.last_action": expected_action} if expected_action else {}
    if expected_action == "PLAN":
        expected_outcome["interaction.plan_created"] = True
    for field, value in SCENARIO_ACTION_STATE_UPDATES.get(scenario_id, {}).get(expected_action, {}).items():
        expected_outcome[f"interaction.{field}"] = value
    return CaseSpec(
        case_id=case_id,
        scenario=scenario_id,
        backend_record={
            "record": {"record_id": record_id, "customer_id": customer_id},
            "private_state": {"system_variables": system_variables},
            "public_state": system_variables,
            "interaction": {
                "last_action": None,
                "status": "open",
                "answer_completed": False,
                "plan_created": False,
            },
        },
        user_goal={"type": user_intent, "desired_action": expected_action},
        user_knowledge={
            "record_id": record_id,
            "customer_id": customer_id,
            "knows_record_id": True,
            "knows_customer_id": True,
        },
        user_policy={
            "mode": user_policy_mode,
            "truthfulness": "truthful" if user_policy_mode == "truthful" else "unreliable",
            "reveal_record_id_on_request": True,
            "reveal_customer_id_on_request": True,
        },
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
            "required_backend_verifications": required_backend_verifications,
            "backend_system_variables": dict(system_variables),
        },
    )


def create_backend(case_spec: CaseSpec) -> BackendEnvironment:
    if case_spec.scenario == "ecommerce_refund":
        return EcommerceBackend(case_spec)
    if case_spec.scenario in {
        "telecom_package",
        "property_service",
        "logistics_delivery",
        "airline_refund",
        "online_education",
    }:
        return ScenarioBackend(case_spec)
    raise ValueError(f"No authoritative backend registered for scenario: {case_spec.scenario}")
