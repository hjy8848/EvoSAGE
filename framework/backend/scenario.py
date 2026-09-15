"""Generic deterministic backend adapters for the five non-ecommerce scenarios."""

from typing import Any, Dict, List
import re

from .base import BackendEnvironment, tool_definition
from .types import ActionResult, CaseSpec, ToolResult


SCENARIO_QUERY_TOOLS = {
    "telecom_package": [
        ("query_package", "查询套餐、合约和当前套餐状态"),
        ("query_account", "查询客户账户状态"),
        ("query_penalty", "查询套餐违约金"),
    ],
    "property_service": [
        ("query_property_account", "查询住户和房屋账户信息"),
        ("query_fee_status", "查询物业缴费状态"),
        ("query_repair_ticket", "查询维修工单状态"),
    ],
    "logistics_delivery": [
        ("query_delivery_order", "查询快递订单和物流状态"),
        ("query_insurance", "查询快递保险信息"),
        ("query_delivery_details", "查询收寄件和配送详情"),
    ],
    "airline_refund": [
        ("query_booking", "查询航班订单和机票状态"),
        ("query_member_profile", "查询会员等级"),
        ("query_ticket_rules", "查询退改签和保险规则"),
        ("query_flight_status", "查询航班状态"),
    ],
    "online_education": [
        ("query_course_enrollment", "查询课程购买和学习状态"),
        ("query_learning_history", "查询历史投诉和学习记录"),
        ("query_user_risk", "查询账户风险状态"),
        ("query_refund_eligibility", "查询课程退款资格"),
        ("search_course_content", "检索课程知识内容"),
    ],
}

SCENARIO_ACTION_TOOLS = {
    "telecom_package": {
        "change_package": "ChangeOrder", "transfer_human": "TransHuman", "end_service": "GoodBye",
    },
    "property_service": {
        "make_payment": "Payment", "register_repair": "Registration", "comfort_resident": "Comfort",
        "transfer_human": "TransHuman", "reject_request": "Reject", "provide_payment_information": "PayInformation",
    },
    "logistics_delivery": {
        "intercept_delivery": "Interception", "modify_delivery": "Modify", "register_claim": "Registration",
        "request_supplement": "Supplementary", "pay_difference": "MakeUpDifference",
        "provide_delivery_detail": "Detail", "compensate_customer": "Compensation",
        "transfer_human": "TransHuman", "reject_request": "Reject", "comfort_customer": "Comfort",
    },
    "airline_refund": {
        "reschedule_or_refund": "RescheduleOrRefund", "reschedule_with_fee": "RescheduleOrRefund+HandlingFee",
        "reschedule_with_compensation": "RescheduleOrRefund+Compensation", "provide_supplement": "Supplementary",
        "compensate_passenger": "Compensation", "comfort_passenger": "Comfort", "transfer_human": "TransHuman",
        "reject_request": "Reject", "answer_enquiry": "Enquiry",
    },
    "online_education": {
        "answer_question": "ANSWER", "guide_user": "GUIDE", "review_repeated_issue": "REVIEW",
        "comfort_student": "COMFORT", "plan_learning_resource": "PLAN", "negotiate_refund": "NEGOTIATE",
        "submit_refund": "REFUND", "transfer_human": "TRANSFER_HUMAN",
    },
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


class ScenarioBackend(BackendEnvironment):
    """Deterministic backend that preserves each scenario's existing variables."""

    def __init__(self, case_spec: CaseSpec):
        self._query_tools = SCENARIO_QUERY_TOOLS.get(case_spec.scenario, [])
        self._action_tools = SCENARIO_ACTION_TOOLS.get(case_spec.scenario, {})
        super().__init__(case_spec)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        definitions = []
        identity = {
            "record_id": {"type": "string", "description": "业务记录编号"},
        }
        for name, description in self._query_tools:
            definitions.append(tool_definition(name, description, identity))
        for name, action in self._action_tools.items():
            definitions.append(tool_definition(name, f"执行{action}业务动作", identity))
        return definitions

    def get_action_tool_map(self) -> Dict[str, str]:
        return dict(self._action_tools)

    def _record(self) -> Dict[str, Any]:
        return self.state.get("record", {})

    def _verify_record(self, arguments: Dict[str, Any], require_selected: bool = False) -> bool:
        record_id = arguments.get("record_id") or self.selected_records.get("record_id")
        if record_id != self._record().get("record_id"):
            return False
        if require_selected and self.selected_records.get("record_id") != record_id:
            return False
        self.selected_records["record_id"] = record_id
        return True

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        if not self._verify_record(arguments):
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error_code="record_not_found",
                error_message="未找到有效业务记录，请核对编号。",
            )
        if tool_name not in dict(self._query_tools):
            return super()._execute_tool(tool_name, arguments)
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data={
                "record_id": self._record().get("record_id"),
                "customer_id": self._record().get("customer_id"),
                "system_info": dict(self.state.get("system_info", {})),
                "status": dict(self.state.get("status", {})),
            },
        )

    def _execute_action(self, action_name: str, arguments: Dict[str, Any]) -> ActionResult:
        if not self._verify_record(arguments, require_selected=True):
            return ActionResult(
                success=False,
                action_name=action_name,
                error_code="record_not_verified",
                error_message="执行动作前必须先查询有效业务记录。",
            )
        interaction = self.state.setdefault("interaction", {})
        previous = interaction.get("last_action")
        interaction["last_action"] = action_name
        interaction["status"] = "closed" if action_name in {"GoodBye", "END"} else "processed"
        interaction["repeated"] = previous == action_name
        if action_name == "ANSWER":
            interaction["answer_completed"] = True
        if action_name == "PLAN":
            interaction["plan_created"] = True
        if action_name in {"TransHuman", "TRANSFER_HUMAN"}:
            interaction["escalation_status"] = "Requested"
        self.state.setdefault("action_status", {})[_safe_name(action_name)] = "completed"
        return ActionResult(
            success=True,
            action_name=action_name,
            data={
                "record_id": self._record().get("record_id"),
                "last_action": action_name,
                "repeated": interaction["repeated"],
            },
        )
