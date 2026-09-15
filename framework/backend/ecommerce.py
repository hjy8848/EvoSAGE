"""Synthetic but authoritative ecommerce backend for the MVP migration."""

from typing import Any, Dict, List

from .base import BackendEnvironment, tool_definition
from .types import ActionResult, CaseSpec, ToolResult


class EcommerceBackend(BackendEnvironment):
    scenario_id = "ecommerce_refund"

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        definitions = [
            tool_definition(
                "query_order",
                "查询订单的物流、支付和售后状态。需要用户提供准确订单号。",
                {
                    "order_id": {"type": "string", "description": "订单号"},
                },
                ["order_id"],
            ),
            tool_definition(
                "query_customer_profile",
                "查询客户可用于售后决策的账户等级。",
                {
                    "customer_id": {"type": "string", "description": "客户ID"},
                },
                ["customer_id"],
            ),
            tool_definition(
                "query_payment",
                "查询订单支付状态。",
                {
                    "order_id": {"type": "string", "description": "订单号"},
                },
                ["order_id"],
            ),
        ]
        action_properties = {
            "order_id": {"type": "string", "description": "已核验的订单号"},
        }
        for tool_name, description in {
            "submit_refund": "提交退款申请",
            "intercept_shipment": "申请拦截物流",
            "exchange_order": "提交换货申请",
            "schedule_pickup": "安排上门取件",
            "request_document": "要求补充售后凭证",
            "charge_fee": "要求支付换货运费",
            "comfort_customer": "记录安抚处理",
            "comfort_and_compensate": "记录安抚和赔偿处理",
            "reject_request": "拒绝售后请求",
        }.items():
            definitions.append(tool_definition(tool_name, description, action_properties))
        return definitions

    def get_action_tool_map(self) -> Dict[str, str]:
        return {
            "submit_refund": "Refund",
            "intercept_shipment": "Interception",
            "exchange_order": "Exchange",
            "schedule_pickup": "CollectionService",
            "request_document": "Supplementary",
            "charge_fee": "PayFee",
            "comfort_customer": "Comfort",
            "comfort_and_compensate": "Comfort+Compensation",
            "reject_request": "Reject",
        }

    def _order(self) -> Dict[str, Any]:
        return self.state.get("order", self.state)

    def _customer(self) -> Dict[str, Any]:
        return self.state.get("customer", {})

    def _selected_order(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        order_id = arguments.get("order_id") or self.selected_records.get("order_id")
        order = self._order()
        if not order_id or order.get("order_id") != order_id:
            return {}
        self.selected_records["order_id"] = order_id
        return order

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        if tool_name == "query_order":
            order = self._selected_order(arguments)
            if not order:
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error_code="order_not_found",
                    error_message="未找到该订单，请核对订单号。",
                )
            # 只返回客服有权限看到的字段，不返回完整 backend_record。
            return ToolResult(
                success=True,
                tool_name=tool_name,
                data={
                    "order_id": order.get("order_id"),
                    "customer_id": self._customer().get("customer_id"),
                    "product": order.get("product"),
                    "shipping_status": order.get("shipping_status"),
                    "payment_status": order.get("payment_status"),
                    "refund_status": order.get("refund_status"),
                    "refund_eligible": order.get("refund_eligible"),
                    "return_window_open": order.get("return_window_open"),
                },
            )

        if tool_name == "query_customer_profile":
            customer_id = arguments.get("customer_id")
            customer = self._customer()
            if not customer_id or customer.get("customer_id") != customer_id:
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error_code="customer_not_found",
                    error_message="未找到该客户账户。",
                )
            self.selected_records["customer_id"] = customer_id
            return ToolResult(
                success=True,
                tool_name=tool_name,
                data={"customer_id": customer_id, "credit_level": customer.get("credit_level")},
            )

        if tool_name == "query_payment":
            order = self._selected_order(arguments)
            if not order:
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error_code="order_not_found",
                    error_message="未找到该订单，请核对订单号。",
                )
            return ToolResult(
                success=True,
                tool_name=tool_name,
                data={
                    "order_id": order.get("order_id"),
                    "payment_status": order.get("payment_status"),
                    "paid_amount": order.get("paid_amount"),
                },
            )

        return super()._execute_tool(tool_name, arguments)

    def _execute_action(self, action_name: str, arguments: Dict[str, Any]) -> ActionResult:
        requested_order_id = arguments.get("order_id") or self.selected_records.get("order_id")
        if self.selected_records.get("order_id") != requested_order_id:
            return ActionResult(
                success=False,
                action_name=action_name,
                error_code="order_not_verified",
                error_message="执行售后动作前必须先查询有效订单。",
            )
        order = self._selected_order(arguments)
        if not order:
            return ActionResult(
                success=False,
                action_name=action_name,
                error_code="order_not_verified",
                error_message="执行售后动作前必须先查询有效订单。",
            )

        shipping_status = order.get("shipping_status")
        refund_eligible = order.get("refund_eligible", False)

        if action_name in {"Refund", "submit_refund"}:
            if shipping_status != "Unshipped":
                return ActionResult(
                    success=False,
                    action_name=action_name,
                    error_code="refund_requires_review",
                    error_message="订单不是未发货状态，不能直接退款。",
                )
            if not refund_eligible:
                return ActionResult(
                    success=False,
                    action_name=action_name,
                    error_code="refund_not_eligible",
                    error_message="该订单不满足直接退款条件。",
                )
            order["refund_status"] = "Approved"
            order["last_action"] = action_name
            return ActionResult(
                success=True,
                action_name=action_name,
                data={"order_id": order.get("order_id"), "refund_status": "Approved"},
            )

        if action_name in {"Interception", "intercept_shipment"}:
            if shipping_status != "Shipping":
                return ActionResult(
                    success=False,
                    action_name=action_name,
                    error_code="interception_unavailable",
                    error_message="当前物流状态不支持拦截。",
                )
            order["interception_status"] = "Requested"
            order["last_action"] = action_name
            return ActionResult(
                success=True,
                action_name=action_name,
                data={"order_id": order.get("order_id"), "interception_status": "Requested"},
            )

        if action_name in {"CollectionService", "schedule_pickup"}:
            if shipping_status != "Signed":
                return ActionResult(
                    success=False,
                    action_name=action_name,
                    error_code="pickup_unavailable",
                    error_message="当前订单状态不支持上门取件。",
                )
            order["return_status"] = "PickupScheduled"
            order["last_action"] = action_name
            return ActionResult(
                success=True,
                action_name=action_name,
                data={"order_id": order.get("order_id"), "return_status": "PickupScheduled"},
            )

        if action_name in {"Exchange", "exchange_order"}:
            order["exchange_status"] = "Approved"
            order["last_action"] = action_name
            return ActionResult(
                success=True,
                action_name=action_name,
                data={"order_id": order.get("order_id"), "exchange_status": "Approved"},
            )

        if action_name in {"Supplementary", "request_document"}:
            order["documents_requested"] = True
            order["last_action"] = action_name
            return ActionResult(
                success=True,
                action_name=action_name,
                data={"order_id": order.get("order_id"), "documents_requested": True},
            )

        if action_name == "Reject":
            order["refund_status"] = "Rejected"
            order["last_action"] = action_name
            return ActionResult(
                success=True,
                action_name=action_name,
                data={"order_id": order.get("order_id"), "refund_status": "Rejected"},
            )

        # 安抚、补偿、人工转接等动作先记录业务动作，暂不改变订单状态。
        order["last_action"] = action_name
        return ActionResult(
            success=True,
            action_name=action_name,
            data={"order_id": order.get("order_id"), "recorded": True},
        )
