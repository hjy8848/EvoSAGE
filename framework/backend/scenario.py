"""Deterministic authoritative backend adapters for the non-ecommerce scenarios."""

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

# Only these fields may be returned by a query.  The complete backend record
# remains evaluator-only; a tool never returns the private state wholesale.
SCENARIO_PUBLIC_FIELDS = {
    "telecom_package": {
        "query_package": ("PackageStatus", "package_status"),
        "query_account": ("AccountStatus", "account_status"),
        "query_penalty": ("Penalty", "penalty"),
    },
    "property_service": {
        "query_property_account": ("HouseStatus", "house_status"),
        "query_fee_status": ("FeePaymentStatus", "fee_payment_status"),
        "query_repair_ticket": ("RepairTicketStatus", "repair_ticket_status"),
    },
    "logistics_delivery": {
        "query_delivery_order": ("orderStatus", "order_status"),
        "query_insurance": ("hasInsurance", "has_insurance"),
        "query_delivery_details": ("deliveryAddress", "delivery_address"),
    },
    "airline_refund": {
        "query_booking": ("BookingStatus", "booking_status"),
        "query_member_profile": ("memberLevel", "member_level"),
        "query_ticket_rules": ("hasInsurance", "has_insurance"),
        "query_flight_status": ("flightStatus", "flight_status"),
    },
    "online_education": {
        "query_course_enrollment": ("CourseStatus", "course_status"),
        "query_learning_history": ("HistoricalComplaintRecords", "historical_complaints"),
        "query_user_risk": ("isRiskUser", "is_risk_user"),
        "query_refund_eligibility": ("RefundEligibility", "refund_eligibility"),
        "search_course_content": ("KnowledgeResources", "knowledge_resources"),
    },
}

SCENARIO_ACTION_STATE_UPDATES = {
    "telecom_package": {
        "ChangeOrder": {"package_status": "Changed"},
        "TransHuman": {"escalation_status": "Requested"},
        "GoodBye": {"interaction_status": "Closed"},
    },
    "property_service": {
        "Payment": {"fee_payment_status": "Paid"},
        "Registration": {"repair_ticket_status": "Registered"},
        "TransHuman": {"escalation_status": "Requested"},
        "Reject": {"request_status": "Rejected"},
        "PayInformation": {"information_status": "Provided"},
    },
    "logistics_delivery": {
        "Interception": {"interception_status": "Requested"},
        "Modify": {"delivery_modification_status": "Requested"},
        "Registration": {"claim_status": "Registered"},
        "Supplementary": {"supplement_status": "Requested"},
        "MakeUpDifference": {"difference_status": "Paid"},
        "Detail": {"delivery_detail_status": "Provided"},
        "Compensation": {"compensation_status": "Approved"},
        "TransHuman": {"escalation_status": "Requested"},
        "Reject": {"request_status": "Rejected"},
        "Comfort": {"comfort_status": "Recorded"},
    },
    "airline_refund": {
        "RescheduleOrRefund": {"booking_action_status": "Completed"},
        "RescheduleOrRefund+HandlingFee": {"booking_action_status": "CompletedWithFee"},
        "RescheduleOrRefund+Compensation": {"booking_action_status": "CompletedWithCompensation"},
        "Supplementary": {"supplement_status": "Requested"},
        "Compensation": {"compensation_status": "Approved"},
        "Comfort": {"comfort_status": "Recorded"},
        "TransHuman": {"escalation_status": "Requested"},
        "Reject": {"request_status": "Rejected"},
        "Enquiry": {"enquiry_status": "Answered"},
    },
    "online_education": {
        "ANSWER": {"answer_completed": True},
        "GUIDE": {"guidance_status": "Provided"},
        "REVIEW": {"review_status": "Opened"},
        "COMFORT": {"comfort_status": "Recorded"},
        "PLAN": {"plan_created": True, "resource_status": "Allocated"},
        "NEGOTIATE": {"negotiation_status": "Opened"},
        "REFUND": {"refund_status": "Submitted"},
        "TRANSFER_HUMAN": {"escalation_status": "Requested"},
    },
}

# State changes are applied to the authoritative public projection as well as
# the interaction audit record.  This makes a second query observe the
# business outcome instead of only seeing ``last_action`` in an evaluator
# side-channel.
SCENARIO_ACTION_PUBLIC_STATE_UPDATES = {
    "telecom_package": {
        "ChangeOrder": {"PackageStatus": "Changed", "CurrentPlan": "Changed"},
    },
    "property_service": {
        "Payment": {"FeePaymentStatus": "Paid"},
        "Registration": {"RepairTicketStatus": "Registered"},
    },
    "logistics_delivery": {
        "Interception": {"orderStatus": "InterceptionRequested"},
        "Modify": {"orderStatus": "ModificationRequested"},
    },
    "airline_refund": {
        "RescheduleOrRefund": {"BookingStatus": "Rescheduled"},
        "RescheduleOrRefund+HandlingFee": {"BookingStatus": "RescheduledWithFee"},
        "RescheduleOrRefund+Compensation": {"BookingStatus": "RescheduledWithCompensation"},
    },
    "online_education": {
        "REFUND": {"RefundEligibility": False},
        "PLAN": {"ResourceStatus": "Allocated"},
    },
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


class ScenarioBackend(BackendEnvironment):
    """Scenario-specific public projections over a hidden deterministic state."""

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
        public_field = SCENARIO_PUBLIC_FIELDS.get(self.case_spec.scenario, {}).get(tool_name)
        if public_field is None:
            return super()._execute_tool(tool_name, arguments)
        backend_key, public_key = public_field
        public_state = self.state.get("public_state", {})
        data = {
            "record_id": self._record().get("record_id"),
            "customer_id": self._record().get("customer_id"),
            public_key: public_state.get(backend_key),
        }
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data=data,
        )

    def _execute_action(self, action_name: str, arguments: Dict[str, Any]) -> ActionResult:
        if not self._verify_record(arguments, require_selected=True):
            return ActionResult(
                success=False,
                action_name=action_name,
                error_code="record_not_verified",
                error_message="执行动作前必须先查询有效业务记录。",
            )

        precondition_error = self._action_precondition_error(action_name)
        if precondition_error is not None:
            error_code, error_message = precondition_error
            return ActionResult(
                success=False,
                action_name=action_name,
                error_code=error_code,
                error_message=error_message,
            )

        interaction = self.state.setdefault("interaction", {})
        previous = interaction.get("last_action")
        interaction["last_action"] = action_name
        interaction["status"] = "closed" if action_name in {"GoodBye", "END"} else "processed"
        interaction["repeated"] = previous == action_name
        updates = SCENARIO_ACTION_STATE_UPDATES.get(self.case_spec.scenario, {}).get(action_name, {})
        interaction.update(updates)
        self.state.setdefault("action_status", {})[_safe_name(action_name)] = "completed"
        public_updates = SCENARIO_ACTION_PUBLIC_STATE_UPDATES.get(
            self.case_spec.scenario, {}
        ).get(action_name, {})
        if public_updates:
            self.state.setdefault("public_state", {}).update(public_updates)
            self.state.setdefault("private_state", {}).setdefault(
                "system_variables", {}
            ).update(public_updates)
        return ActionResult(
            success=True,
            action_name=action_name,
            data={
                "record_id": self._record().get("record_id"),
                "last_action": action_name,
                "repeated": interaction["repeated"],
                **updates,
                **public_updates,
            },
        )

    def _action_precondition_error(self, action_name: str):
        """Return an error for a state-invalid action, otherwise ``None``.

        These checks deliberately use only authoritative backend state.  They
        do not inspect the Agent's declared path or final action, so a wrong
        action cannot become valid merely because it matches hidden gold.
        """
        public = self.state.get("public_state", {})
        scenario = self.case_spec.scenario

        def fail(code, message):
            return code, message

        if scenario == "telecom_package":
            if action_name == "ChangeOrder" and public.get("AccountStatus") != "Active":
                return fail("account_inactive", "账户当前状态不支持变更套餐。")

        elif scenario == "property_service":
            fee_status = public.get("FeePaymentStatus")
            if action_name in {"Payment", "Reject"} and fee_status != "Unpaid":
                return fail("payment_action_unavailable", "当前物业缴费状态不支持该动作。")
            if action_name == "Registration" and fee_status == "Unpaid":
                return fail("repair_requires_settled_fees", "物业费未结清，当前不能登记该维修工单。")

        elif scenario == "logistics_delivery":
            order_status = public.get("orderStatus")
            insured = public.get("hasInsurance")
            if action_name == "Interception" and order_status == "Delivered":
                return fail("interception_unavailable", "包裹已送达，当前不能申请拦截。")
            if action_name == "Modify" and order_status != "Undelivered":
                return fail("modification_unavailable", "只有未送达订单可以修改配送信息。")
            if action_name == "MakeUpDifference" and order_status != "Delivered":
                return fail("difference_payment_unavailable", "只有已送达订单可以支付改派差额。")
            if action_name == "Registration" and order_status not in {"Delivered", "Undelivered"}:
                return fail("claim_unavailable", "当前物流状态不能登记理赔或异常工单。")
            if action_name == "Compensation" and not (
                order_status == "Arrived" and insured is True
            ):
                return fail("compensation_unavailable", "只有已到达且已投保的包裹才能直接赔付。")
            if action_name == "TransHuman" and not (
                order_status == "Arrived" and insured is False
            ):
                return fail("escalation_unavailable", "当前物流状态不满足该升级条件。")
            if action_name in {"Reject", "Comfort"} and order_status != "Arrived":
                return fail("complaint_action_unavailable", "当前物流状态不支持该投诉处理动作。")

        elif scenario == "airline_refund":
            member_level = public.get("memberLevel")
            insured = public.get("hasInsurance")
            if action_name == "RescheduleOrRefund+HandlingFee" and not (
                member_level == "Regular" and insured is False
            ):
                return fail("handling_fee_path_unavailable", "当前会员或保险状态不支持收取改签手续费。")
            if action_name == "RescheduleOrRefund+Compensation" and member_level != "VIP":
                return fail("compensation_reschedule_unavailable", "只有 VIP 会员满足该改签赔偿条件。")
            if action_name == "Compensation" and member_level != "Regular":
                return fail("passenger_compensation_unavailable", "当前会员状态不支持该赔偿动作。")
            if action_name == "TransHuman" and member_level != "VIP":
                return fail("airline_escalation_unavailable", "当前会员状态不满足该升级条件。")
            if action_name == "Reject" and member_level != "Blacklist":
                return fail("airline_rejection_unavailable", "当前会员状态不支持拒绝该请求。")

        elif scenario == "online_education":
            risk_user = public.get("isRiskUser")
            refund_eligible = public.get("RefundEligibility")
            resources = public.get("KnowledgeResources")
            if action_name == "NEGOTIATE" and risk_user is not True:
                return fail("negotiation_unavailable", "当前账户风险状态不支持协商退款。")
            if action_name == "REFUND" and not (
                risk_user is False and refund_eligible is True
            ):
                return fail("refund_unavailable", "当前账户状态或课程资格不支持退款。")
            if action_name == "PLAN" and not (
                risk_user is False and resources
            ):
                return fail("resource_plan_unavailable", "当前账户状态或课程资源不支持制定学习计划。")

        return None
