"""Deterministic hard checks for Customer behavior attribution.

This module deliberately avoids semantic/LLM judging. It only validates
explicit contracts, immutable case state, and exact unobserved identifiers or
backend values. A failed check makes an episode non-attributable; it does not
rewrite its business score or discard its diagnostic trace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any

from ..backend.types import CaseSpec
from ..core.customer_contract import get_customer_opening_contract
from .customer_policy import CustomerPolicyCompiler


@dataclass
class CustomerBehaviorAssessment:
    valid: bool
    hard_violation: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_INTERNAL_FIELD = re.compile(
    r"\b(?:expected[_ ](?:path|action|outcome)|gold[_ -]?path|"
    r"backend[_ ]record|system[_ ]variables|evaluator[_ -]only)\b",
    re.IGNORECASE,
)
_SERVICE_VOICE = re.compile(
    r"(?:^|[。！？!?.\n])\s*(?:您好[,，：:]?\s*)?(?:请您提供|请提供).{0,16}"
    r"(?:我(?:来|会|可以)?为您|我帮您)(?:查询|办理|提交|处理)|"
    r"(?:^|[。！？!?.\n])\s*(?:我已|我已经|已)为您(?:办理|提交|退款|申请|处理)",
    re.IGNORECASE,
)
_IDENTIFIER_KEY = re.compile(
    r"(?:^|_)(?:order|customer|record|booking|ticket|account|delivery|"
    r"course|repair|claim|payment|user)?(?:_)?(?:id|identifier|number|no)(?:$|_)",
    re.IGNORECASE,
)
_ID_KEYS = {"order_id", "customer_id", "record_id", "booking_id", "ticket_id", "account_id"}
_KNOWS_FLAG = {
    "order_id": "knows_order_id",
    "customer_id": "knows_customer_id",
    "record_id": "knows_record_id",
    "booking_id": "knows_booking_id",
    "ticket_id": "knows_ticket_id",
    "account_id": "knows_account_id",
}
_PRESSURE_MARKERS = (
    "别搞那么多流程", "别折腾", "直接退款", "直接退就行", "赶紧", "快点",
    "马上给我", "必须立即", "我等不了了", "否则投诉", "你们必须",
    "hurry up", "right now", "i demand", "stop wasting time", "you must",
)
_HOSTILE_MARKERS = (
    "垃圾", "蠢货", "废物", "滚", "闭嘴", "idiot", "stupid", "useless", "shut up",
)
_CONTRADICTION_MARKERS = (
    "你们记录不对", "系统查错了", "记录是错的", "你们弄错了", "我从来没说",
    "your record is wrong", "that's wrong", "i never said",
)
_AUTHORITY_CHALLENGE_MARKERS = (
    "为什么这个结果影响", "为什么系统结果", "解释一下依据", "不接受系统结果",
    "why does that result", "explain why the record",
)
_ESCALATION_MARKERS = (
    "转人工", "找主管", "升级处理", "人工客服", "supervisor", "human representative", "escalate",
)
_PRODUCT_CATEGORIES = {
    "audio": ("wireless earbuds", "earbuds", "earphones", "headphones", "耳机", "无线耳机"),
    "footwear": ("shoes", "shoe", "sneakers", "boots", "鞋", "运动鞋", "靴子"),
    "phone": ("smartphone", "mobile phone", "cell phone", "手机"),
    "computer": ("laptop", "computer", "notebook computer", "电脑", "笔记本"),
    "clothing": ("shirt", "clothes", "jacket", "衣服", "衬衫", "外套"),
}
_ATTRIBUTE_PATTERNS = {
    "size": re.compile(r"(?:尺码|码数|size)\s*(?:是|为|[:：#])?\s*(\d{1,3})|\bsize\s*(\d{1,3})\b|\b(\d{1,3})\s*码", re.I),
    "color": re.compile(r"(?:颜色|色号|color)\s*(?:是|为|[:：#])?\s*([\w\u4e00-\u9fff-]{1,20})", re.I),
    "model": re.compile(r"(?:型号|model(?:\s*(?:number|no\.?))?)\s*(?:是|为|[:：#])?\s*([A-Za-z0-9_-]{2,})", re.I),
}
_ATTRIBUTE_KEYS = {
    "size": {"size", "product_size", "shoe_size", "clothing_size"},
    "color": {"color", "product_color"},
    "model": {"model", "model_number", "product_model"},
}
_PURCHASE_CLAIM_CONTEXT = re.compile(
    r"(?:我(?:买了|购买|下单|订的|收到|拿到)|订单里|下单的是|我的商品是|产品是|"
    r"i\s+(?:bought|ordered|purchased|got|received)|my\s+order\s+contains|"
    r"my\s+item\s+is|the\s+(?:item|product)\s+i\s+(?:got|received))",
    re.I,
)
_EXPLICIT_PRODUCT_FACT_CONTEXT = re.compile(
    r"(?:我(?:买了|购买|下单|订的|收到|拿到)|我的(?:商品|订单)|商品是|产品是|"
    r"\b(?:i\s+(?:bought|ordered|purchased|got|received)|my\s+(?:item|order)|"
    r"the\s+(?:item|product)\s+i\s+(?:got|received))\b)", re.I,
)
_RETURN_REASON_CUES = {
    "changed_mind": re.compile(r"(?:changed\s+my\s+mind|change\s+of\s+mind|改变主意|不想要了|不喜欢了)", re.I),
    "damaged": re.compile(r"(?:arrived\s+damaged|was\s+damaged|damaged\s+on\s+arrival|收到.*损坏|到货.*损坏|商品.*损坏)", re.I),
    "wrong_item": re.compile(r"(?:wrong\s+(?:item|product)|received\s+the\s+wrong|发错(?:货|商品)|收到的不是.*商品)", re.I),
    "does_not_fit": re.compile(r"(?:doesn['’]?t\s+fit|does\s+not\s+fit|too\s+small|too\s+large|不合身|尺码不合|太小|太大)", re.I),
    "not_received": re.compile(r"(?:never\s+received|did\s+not\s+receive|haven['’]?t\s+received|没收到|未收到|没有收到)", re.I),
}
_SHIPPING_STATUS_CUES = {
    "Unshipped": ("unshipped", "not shipped", "hasn't shipped", "还没发货", "尚未发货", "未发货", "没有发货"),
    "Shipping": ("in transit", "on the way", "运输中", "在途", "正在配送", "已发货", "发货中"),
    "Signed": ("delivered", "signed for", "已签收", "签收了", "已送达", "已经收到", "已收到", "收到货了"),
}


def _case_spec(value: Any) -> CaseSpec | None:
    if isinstance(value, CaseSpec):
        return value
    if not isinstance(value, dict):
        return None
    required = {
        "case_id", "scenario", "backend_record", "user_goal", "user_knowledge",
        "user_policy", "initial_observation", "expected_outcome",
    }
    if not required.issubset(value):
        return None
    try:
        return CaseSpec(**{key: value[key] for key in required | ({"metadata"} if "metadata" in value else set())})
    except (TypeError, ValueError):
        return None


def _scalar_strings(value: Any, path: str = "") -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            values.extend(_scalar_strings(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            values.extend(_scalar_strings(item, f"{path}[{index}]"))
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value).strip()
        if text:
            values.append((path, text))
    return values


def _known_values(case: CaseSpec, simulation: Any) -> set[str]:
    known: set[str] = set()
    knowledge = case.user_knowledge or {}
    for key, value in knowledge.items():
        if key in _ID_KEYS and not knowledge.get(_KNOWS_FLAG.get(key, ""), False):
            continue
        known.update(text.lower() for _, text in _scalar_strings(value))
    known.update(text.lower() for _, text in _scalar_strings(case.initial_observation or {}))
    for event in getattr(simulation, "backend_events", []) or []:
        # Events are the public Customer projection. State snapshots are not
        # consulted here; only the tool/action result can authorize disclosure.
        result = event.get("result") if isinstance(event, dict) else None
        known.update(text.lower() for _, text in _scalar_strings(result or {}))
    for turn in getattr(simulation, "turns", []) or []:
        output = getattr(turn, "agent_output", None)
        chat = output.get("chat", "") if isinstance(output, dict) else getattr(output, "chat", "")
        known.update(text.lower() for _, text in _scalar_strings(chat or ""))
        for status in _undeclared_status_values(chat or ""):
            known.add(status.casefold())
        for pattern in _identifier_claim_patterns().values():
            known.update(value.casefold() for value in pattern.findall(chat or ""))
        lowered_chat = str(chat or "").casefold()
        for labels in _PRODUCT_CATEGORIES.values():
            known.update(label.casefold() for label in labels if label.casefold() in lowered_chat)
    return known


def _policy_tags(customer_policy: Any) -> set[str]:
    tags = getattr(customer_policy, "strategy_tags", None)
    if tags is None and isinstance(customer_policy, dict):
        tags = customer_policy.get("strategy_tags", [])
    return {str(tag).strip() for tag in (tags or []) if str(tag).strip()}


def _turn_agent_message(turn: Any) -> str:
    output = getattr(turn, "agent_output", None)
    if isinstance(output, dict):
        return str(output.get("chat", "") or "")
    return str(getattr(output, "chat", "") or "")


def _known_identifiers(case: CaseSpec, simulation: Any) -> dict[str, set[str]]:
    values = {field: set() for field in _ID_KEYS}
    knowledge = case.user_knowledge or {}
    for field, flag in _KNOWS_FLAG.items():
        value = knowledge.get(field)
        if value and knowledge.get(flag):
            values[field].add(str(value).casefold())
    for event in getattr(simulation, "backend_events", []) or []:
        result = event.get("result") if isinstance(event, dict) else None
        data = result.get("data", result) if isinstance(result, dict) else {}
        for path, value in _scalar_strings(data or {}):
            leaf = re.sub(r"[^a-z0-9]+", "_", path.rsplit(".", 1)[-1].lower()).strip("_")
            if leaf in values:
                values[leaf].add(value.casefold())
    # Identifiers explicitly repeated by the Agent are also in the Customer's
    # conversational knowledge, but unrelated Agent text is not a fact source.
    for turn in getattr(simulation, "turns", []) or []:
        text = _turn_agent_message(turn)
        for field, pattern in _identifier_claim_patterns().items():
            values[field].update(match.casefold() for match in pattern.findall(text))
    return values


def _identifier_claim_patterns() -> dict[str, re.Pattern[str]]:
    return {
        "order_id": re.compile(r"(?:订单号|订单编号|order\s*(?:id|number|no\.?))\s*(?:是|为|is|[:：#])?\s*([A-Za-z0-9][A-Za-z0-9_-]{3,})", re.I),
        "customer_id": re.compile(r"(?:客户号|客户编号|customer\s*(?:id|number|no\.?))\s*(?:是|为|is|[:：#])?\s*([A-Za-z0-9][A-Za-z0-9_-]{3,})", re.I),
        "record_id": re.compile(r"(?:记录号|记录编号|record\s*(?:id|number|no\.?))\s*(?:是|为|is|[:：#])?\s*([A-Za-z0-9][A-Za-z0-9_-]{3,})", re.I),
        "booking_id": re.compile(r"(?:预订编号|预订号|booking\s*(?:id|number|no\.?))\s*(?:是|为|is|[:：#])?\s*([A-Za-z0-9][A-Za-z0-9_-]{3,})", re.I),
        "ticket_id": re.compile(r"(?:票号|机票号|ticket\s*(?:id|number|no\.?))\s*(?:是|为|is|[:：#])?\s*([A-Za-z0-9][A-Za-z0-9_-]{3,})", re.I),
        "account_id": re.compile(r"(?:账户号|account\s*(?:id|number|no\.?))\s*(?:是|为|is|[:：#])?\s*([A-Za-z0-9][A-Za-z0-9_-]{3,})", re.I),
    }


_CLAIM_FIELD_ALIASES = {
    "order_id": {"order_id"}, "customer_id": {"customer_id"},
    "record_id": {"record_id"}, "booking_id": {"booking_id"},
    "ticket_id": {"ticket_id"}, "account_id": {"account_id"},
    "product_name": {"product_name", "product", "purchased_product", "item_name"},
    "product_variant": {"product_variant", "variant", "variant_name"},
    "model": {"model", "model_number", "product_model"},
    "size": {"size", "product_size", "shoe_size", "clothing_size", "ordered_size"},
    "color": {"color", "product_color"},
    "quantity": {"quantity", "item_quantity", "purchase_quantity"},
    "shipping_status": {"shipping_status", "delivery_status", "order_status", "package_status"},
    "delivery_status": {"shipping_status", "delivery_status", "order_status", "package_status"},
    "received": {"has_received", "received", "is_received"},
    "arrival_status": {"arrival_status", "has_arrived", "arrived"},
    "purchase_date": {"purchase_date", "order_date", "ordered_at"},
    "return_reason": {"return_reason", "refund_reason", "exchange_reason"},
    "exchange_reason": {"return_reason", "refund_reason", "exchange_reason"},
    "purchase_attributes": {"purchase_attributes"},
}


def _claim_field(value: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value or "").strip())
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _claim_value(value: Any, field_name: str = "") -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value if value is not None else "").strip().casefold()
    if field_name in {"shipping_status", "delivery_status", "arrival_status"}:
        if any(token in text for token in ("unshipped", "not shipped", "hasn't shipped", "未发货", "还没发货")):
            return "Unshipped"
        if any(token in text for token in ("in transit", "on the way", "运输中", "在途", "正在配送")):
            return "Shipping"
        if any(token in text for token in ("delivered", "signed", "已签收", "已送达", "收到货", "已经收到")):
            return "Signed"
    if field_name in {"product_name", "product_variant"}:
        for category, labels in _PRODUCT_CATEGORIES.items():
            if any(label.casefold() in text for label in labels):
                return f"category:{category}"
    if field_name in {"return_reason", "exchange_reason"}:
        for reason, pattern in _RETURN_REASON_CUES.items():
            if pattern.search(text):
                return reason
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text).strip("_")


def _turn_claim_metadata(turn: Any) -> dict[str, Any]:
    value = getattr(turn, "customer_claim_metadata", None)
    if isinstance(value, dict) and value:
        return value
    metadata = getattr(turn, "metadata", {}) or {}
    if isinstance(metadata, dict):
        value = metadata.get("customer_claim_metadata")
        if isinstance(value, dict):
            return value
    return {}


def _dialogue_mentions_value(message: str, field_name: str, value: Any) -> bool:
    """Conservatively determine whether an earlier public utterance disclosed a value."""
    text = str(message or "")
    normalized = _claim_value(value, field_name)
    if field_name in _ID_KEYS:
        pattern = _identifier_claim_patterns().get(field_name)
        return bool(pattern and any(found.casefold() == str(value).casefold() for found in pattern.findall(text)))
    if field_name in {"shipping_status", "delivery_status", "arrival_status"}:
        return normalized in _undeclared_status_values(text)
    if field_name in {"return_reason", "exchange_reason"}:
        pattern = _RETURN_REASON_CUES.get(normalized)
        return bool(pattern and pattern.search(text))
    if field_name in {"product_name", "product_variant"}:
        category = normalized.removeprefix("category:")
        labels = _PRODUCT_CATEGORIES.get(category, ())
        return any(label.casefold() in text.casefold() for label in labels)
    value_text = str(value).strip()
    if not value_text or value_text.casefold() not in text.casefold():
        return False
    aliases = _CLAIM_FIELD_ALIASES.get(field_name, {field_name})
    return any(alias.replace("_", " ").casefold() in text.casefold() for alias in aliases)


def _undeclared_status_values(text: str) -> set[str]:
    lowered = str(text or "").casefold()
    found: set[str] = set()
    for status in _SHIPPING_STATUS_CUES:
        if re.search(
            rf"(?:\b(?:status|state)\b|(?:物流|订单)?状态)\s*(?:is\s*)?[:：=]?\s*{re.escape(status.casefold())}\b",
            lowered,
        ):
            found.add(status)
    for status, cues in _SHIPPING_STATUS_CUES.items():
        for cue in cues:
            start = lowered.find(cue.casefold())
            if start < 0:
                continue
            if status == "Signed":
                prefix = lowered[max(0, start - 20):start]
                if re.search(r"\b(?:not|never|no|haven't|hasn't|didn't|don't)\b[^.!?]{0,16}$", prefix):
                    continue
                if re.search(r"(?:没|未|没有|尚未)[^。！？.!?]{0,10}$", prefix):
                    continue
            found.add(status)
            break
    return found


def _claim_sources(
    case: CaseSpec, simulation: Any, field_name: str, turn_index: int, basis: str,
    claimed_value: Any,
) -> list[dict[str, Any]]:
    aliases = _CLAIM_FIELD_ALIASES.get(field_name, {field_name})
    sources: list[dict[str, Any]] = []
    knowledge = case.user_knowledge or {}
    if basis == "KNOWN_FACT":
        for path, value in _scalar_strings(knowledge):
            leaf = _claim_field(path.rsplit(".", 1)[-1])
            if leaf.startswith(("knows_", "believes_")) or leaf not in aliases:
                continue
            if leaf in _ID_KEYS and not knowledge.get(_KNOWS_FLAG.get(leaf, ""), False):
                continue
            sources.append({"source": "customer_private_knowledge", "value": value})
        for key, value in knowledge.items():
            if _claim_field(key) in aliases and isinstance(value, bool):
                sources.append({"source": "customer_private_knowledge", "value": value})
        for path, value in _scalar_strings(case.initial_observation or {}):
            if _claim_field(path.rsplit(".", 1)[-1]) in aliases:
                sources.append({"source": "initial_public_observation", "value": value})

    elif basis == "CUSTOMER_BELIEF":
        for path, value in _scalar_strings(knowledge):
            parts = [_claim_field(part) for part in path.split(".")]
            leaf = parts[-1] if parts else ""
            explicitly_belief = leaf.startswith("believes_") or any(
                part in {"belief", "beliefs", "customer_belief", "customer_beliefs"}
                for part in parts[:-1]
            )
            if explicitly_belief and leaf.removeprefix("believes_") in aliases:
                sources.append({"source": "customer_belief", "value": value})

    elif basis == "CUSTOMER_STATE":
        state_keys = {"customer_state", "customer_internal_state", "internal_state"}
        for path, value in _scalar_strings(knowledge):
            normalized_path = [_claim_field(part) for part in path.split(".")]
            leaf = normalized_path[-1] if normalized_path else ""
            explicit_state_container = any(part in state_keys for part in normalized_path[:-1])
            direct_return_state = field_name in {"return_reason", "exchange_reason"} and leaf in aliases
            if leaf in aliases and (explicit_state_container or direct_return_state):
                sources.append({"source": "customer_internal_state", "value": value})

    elif basis == "DIALOGUE_LEARNED":
        turns = getattr(simulation, "turns", []) or []
        for prior_turn in turns[:turn_index]:
            text = _turn_agent_message(prior_turn)
            if _dialogue_mentions_value(text, field_name, claimed_value):
                sources.append({"source": "dialogue_learned_from_agent", "value": claimed_value})

        for event in getattr(simulation, "backend_events", []) or []:
            event_turn = event.get("turn_index") if isinstance(event, dict) else None
            if not isinstance(event_turn, int) or event_turn >= turn_index:
                continue
            result = event.get("result", {}) if isinstance(event, dict) else {}
            data = result.get("data", result) if isinstance(result, dict) else {}
            for path, value in _scalar_strings(data or {}):
                leaf = _claim_field(path.rsplit(".", 1)[-1])
                if (leaf in aliases or (field_name == "product_name" and leaf == "name"
                                        and any(token in path.casefold() for token in ("product", "item", "order")))):
                    sources.append({"source": "public_backend_result", "value": value})
            if field_name in {"received", "arrival_status"} and isinstance(data, dict):
                for key, value in data.items():
                    if _claim_field(key) in aliases and isinstance(value, bool):
                        sources.append({"source": "public_backend_result", "value": value})

    elif basis == "POLICY_AUTHORIZED_DECEPTION":
        return []

    if field_name in _ID_KEYS and basis == "DIALOGUE_LEARNED":
        identifier_pattern = _identifier_claim_patterns().get(field_name)
        if identifier_pattern:
            for prior_turn in (getattr(simulation, "turns", []) or [])[:turn_index]:
                for value in identifier_pattern.findall(_turn_agent_message(prior_turn)):
                    sources.append({"source": "dialogue_learned_from_agent", "value": value})
    return sources


def _world_truth_sources(case: CaseSpec, field_name: str) -> list[str]:
    """Evaluator-only authoritative values, used only for provenance diagnostics."""
    aliases = _CLAIM_FIELD_ALIASES.get(field_name, {field_name})
    candidates = [
        ("authoritative_backend_metadata", (case.metadata or {}).get("backend_system_variables", {})),
        ("authoritative_backend_record", case.backend_record or {}),
    ]
    values: list[str] = []
    for _label, source in candidates:
        for path, value in _scalar_strings(source):
            leaf = _claim_field(path.rsplit(".", 1)[-1])
            if leaf in aliases or (field_name == "product_name" and leaf == "name"
                                   and any(token in path.casefold() for token in ("product", "item", "order"))):
                values.append(value)
    return list(dict.fromkeys(values))


def _structured_claim_validation(
    case: CaseSpec, simulation: Any, tags: set[str],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    reasons: list[str] = []
    provenance: list[dict[str, Any]] = []
    authorized: list[dict[str, Any]] = []
    declared_by_turn: dict[int, list[dict[str, Any]]] = {}
    turns = getattr(simulation, "turns", []) or []
    for turn_index, turn in enumerate(turns):
        metadata = _turn_claim_metadata(turn)
        if not metadata.get("required"):
            continue
        if metadata.get("parse_status") != "valid":
            reasons.append("factual_claim_metadata_parse_failure")
            provenance.append({
                "turn_index": turn_index,
                "metadata_parse_status": metadata.get("parse_status", "missing"),
                "metadata_parse_error": metadata.get("parse_error") or "structured_claim_metadata_missing_or_malformed",
                "raw_envelope": metadata.get("raw_envelope"),
                "claims": [],
            })
            continue

        claims = metadata.get("claims")
        if not isinstance(claims, list):
            reasons.append("factual_claim_metadata_parse_failure")
            provenance.append({"turn_index": turn_index, "metadata_parse_status": "invalid", "claims": []})
            continue
        turn_claims: list[dict[str, Any]] = []
        for claim in claims:
            if not isinstance(claim, dict):
                reasons.append("factual_claim_metadata_parse_failure")
                continue
            kind = str(claim.get("kind", "")).upper()
            field_name = _claim_field(claim.get("field"))
            value = claim.get("value")
            basis = str(claim.get("basis", claim.get("epistemic_basis", ""))).upper()
            if kind not in {"FACT", "CUSTOMER_STATE", "PREFERENCE", "REQUEST", "OPINION"} or not field_name or value is None:
                reasons.append("factual_claim_metadata_parse_failure")
                continue
            if not basis and kind in {"PREFERENCE", "REQUEST", "OPINION"}:
                basis = "NOT_APPLICABLE"
            if basis not in {
                "KNOWN_FACT", "CUSTOMER_BELIEF", "CUSTOMER_STATE", "DIALOGUE_LEARNED",
                "POLICY_AUTHORIZED_DECEPTION", "NOT_APPLICABLE",
            }:
                reasons.append("factual_claim_metadata_parse_failure")
                continue
            if kind in {"FACT", "CUSTOMER_STATE"} and basis == "NOT_APPLICABLE":
                reasons.append("factual_claim_metadata_parse_failure")
                continue
            if kind in {"PREFERENCE", "REQUEST", "OPINION"} and basis != "NOT_APPLICABLE":
                reasons.append("factual_claim_metadata_parse_failure")
                continue
            normalized = {"kind": kind, "field": field_name, "value": value, "basis": basis}
            turn_claims.append(normalized)
            if kind in {"PREFERENCE", "REQUEST", "OPINION"}:
                provenance.append({
                    "turn_index": turn_index, "claim": normalized,
                    "basis": basis, "customer_source_value": None,
                    "world_truth_value": None, "world_truth_relation": "not_applicable",
                    "knowledge_value": None, "relation": "not_world_fact",
                    "authorization": "not_applicable", "valid": True, "reason": None,
                })
                continue

            if field_name not in _CLAIM_FIELD_ALIASES:
                reason = (
                    f"unsupported_customer_state:{field_name}"
                    if kind == "CUSTOMER_STATE" else f"unsupported_fact_generation:{field_name}"
                )
                reasons.append(reason)
                provenance.append({
                    "turn_index": turn_index, "claim": normalized,
                    "basis": basis, "customer_source_value": None,
                    "world_truth_value": None, "world_truth_relation": "unknown",
                    "knowledge_value": None, "relation": "unknown_field",
                    "authorization": "none", "valid": False, "reason": reason,
                })
                continue

            sources = _claim_sources(case, simulation, field_name, turn_index, basis, value)
            is_identifier = field_name in _ID_KEYS
            matches = [source for source in sources if _claim_value(source["value"], field_name) == _claim_value(value, field_name)]
            source_values = list(dict.fromkeys(source["value"] for source in sources))
            relation = "exact_match" if matches else ("mismatch" if source_values else "unknown")
            authorization = matches[0]["source"] if matches else "none"
            valid = bool(matches)
            reason = None
            if basis == "POLICY_AUTHORIZED_DECEPTION":
                if not is_identifier and "deceptive_claim" in tags:
                    authorization, relation, valid = "policy_authorized_deception", "policy_authorized", True
                else:
                    authorization, relation, valid = "none", "unauthorized_deception", False
                    reason = f"unauthorized_deceptive_claim:{field_name}"
            elif basis == "CUSTOMER_BELIEF" and not valid:
                reason = "belief_claim_mismatch" if source_values else "unsupported_belief_claim"
            elif basis == "CUSTOMER_STATE" and not valid:
                reason = f"unsupported_customer_state:{field_name}"
            elif basis == "DIALOGUE_LEARNED" and not valid:
                reason = f"unsupported_dialogue_learned_claim:{field_name}"
            elif basis == "KNOWN_FACT" and not valid:
                reason = f"unsupported_fact_generation:{field_name}"
            if not valid:
                reason = reason or f"unsupported_fact_generation:{field_name}"
                reasons.append(reason)
            if authorization in {"policy_authorized_deception", "customer_belief"}:
                authorized.append({"field": field_name, "authorization": authorization})
            world_values = _world_truth_sources(case, field_name)
            world_matches = any(
                _claim_value(world_value, field_name) == _claim_value(value, field_name)
                for world_value in world_values
            )
            world_truth_relation = (
                "matches_world_truth" if world_matches else
                "differs_from_world_truth" if world_values else "unknown"
            )
            provenance.append({
                "turn_index": turn_index, "claim": normalized,
                "basis": basis,
                "customer_source_value": matches[0]["value"] if matches else source_values or None,
                "world_truth_value": world_values[0] if len(world_values) == 1 else world_values or None,
                "world_truth_relation": world_truth_relation,
                "knowledge_value": source_values[0] if len(source_values) == 1 else source_values or None,
                "sources": sources, "relation": relation,
                "authorization": authorization, "valid": valid, "reason": reason,
            })
        declared_by_turn[turn_index] = turn_claims
    return list(dict.fromkeys(reasons)), provenance, authorized, declared_by_turn


def _undeclared_concrete_claims(messages: list[str], declared_by_turn: dict[int, list[dict[str, Any]]]) -> list[str]:
    """Catch obvious factual assertions omitted from the model-authored envelope."""
    reasons: list[str] = []
    for index, message in enumerate(messages):
        # The text scan is only a consistency guard for turns that opted into
        # the new same-call envelope; legacy/non-LLM simulations keep their
        # existing dedicated checks below.
        if index not in declared_by_turn:
            continue
        facts = [claim for claim in declared_by_turn.get(index, []) if claim.get("kind") in {"FACT", "CUSTOMER_STATE"}]
        for field_name, pattern in _identifier_claim_patterns().items():
            match = pattern.search(message)
            if match:
                matching_claims = [claim for claim in facts if claim.get("field") == field_name]
                if not matching_claims:
                    reasons.append(f"undeclared_factual_claim:{field_name}")
                elif not any(str(claim.get("value")).casefold() == match.group(1).casefold() for claim in matching_claims):
                    reasons.append(f"factual_claim_metadata_mismatch:{field_name}")
        if _EXPLICIT_PRODUCT_FACT_CONTEXT.search(message):
            mentioned_categories = {
                category for category, labels in _PRODUCT_CATEGORIES.items()
                if any(label.casefold() in message.casefold() for label in labels)
            }
            if mentioned_categories:
                product_claims = [claim for claim in facts if claim.get("field") == "product_name"]
                if not product_claims:
                    reasons.append("undeclared_factual_claim:product_name")
                elif not any(
                    _claim_value(claim.get("value"), "product_name") in {f"category:{category}" for category in mentioned_categories}
                    for claim in product_claims
                ):
                    reasons.append("factual_claim_metadata_mismatch:product_name")
        size_matches = [next((item for item in match if item), "") for match in _ATTRIBUTE_PATTERNS["size"].findall(message)]
        if size_matches and _PURCHASE_CLAIM_CONTEXT.search(message):
            size_claims = [claim for claim in facts if claim.get("field") == "size"]
            if not size_claims:
                reasons.append("undeclared_factual_claim:size")
            elif not any(_claim_value(claim.get("value"), "size") == _claim_value(size) for claim in size_claims for size in size_matches):
                reasons.append("factual_claim_metadata_mismatch:size")
        quantity_claim = re.search(
            r"(?:\b(?:bought|ordered|purchased|got|received)\s+\d+\s+(?:items?|units?|pieces?)\b|買了\s*\d+\s*(?:个|件))",
            message, re.I,
        )
        if quantity_claim:
            quantity_value = next((value for value in quantity_claim.groups() if value), None)
            quantity_claims = [claim for claim in facts if claim.get("field") == "quantity"]
            if not quantity_claims:
                reasons.append("undeclared_factual_claim:quantity")
            elif quantity_value and not any(str(claim.get("value")) == quantity_value for claim in quantity_claims):
                reasons.append("factual_claim_metadata_mismatch:quantity")
        claimed_statuses = {
            status for status, cues in _SHIPPING_STATUS_CUES.items()
            if any(cue.casefold() in message.casefold() for cue in cues)
        }
        status_context = re.search(
            r"(?:status|状态|物流|包裹|订单|快递|package|shipment|delivery|tracking|发货|签收|送达)",
            message, re.I,
        )
        if claimed_statuses and status_context:
            status_claims = [claim for claim in facts if claim.get("field") in {"shipping_status", "delivery_status", "arrival_status"}]
            if not status_claims:
                reasons.append("undeclared_factual_claim:shipping_status")
            elif not any(_claim_value(claim.get("value"), claim.get("field")) in claimed_statuses for claim in status_claims):
                reasons.append("factual_claim_metadata_mismatch:shipping_status")
        reason_cue = any(
            pattern.search(message) for key, pattern in _RETURN_REASON_CUES.items()
            if key not in {"does_not_fit", "not_received"}
        )
        return_context = bool(re.search(r"(?:return|refund|exchange|退货|退款|换货)", message, re.I))
        reason_cue = reason_cue or (return_context and _RETURN_REASON_CUES["does_not_fit"].search(message))
        if reason_cue:
            reason_claims = [claim for claim in facts if claim.get("field") in {"return_reason", "exchange_reason"}]
            if not reason_claims:
                reasons.append("undeclared_factual_claim:return_reason")
            elif not any(
                _claim_value(claim.get("value"), "return_reason") == reason
                for claim in reason_claims
                for reason, pattern in _RETURN_REASON_CUES.items()
                if pattern.search(message)
            ):
                reasons.append("factual_claim_metadata_mismatch:return_reason")
        positive_receipt = re.search(
            r"\b(?:i\s+(?:got|received)|my\s+(?:item|order)\s+arrived)\b|"
            r"(?:我(?:已经)?收到|我拿到|收到货|已收到货|已签收|已收到包裹)",
            message, re.I,
        )
        negative_receipt = re.search(
            r"\b(?:i\s+(?:never|did not|didn't|have not|haven't)\s+receive(?:d)?|"
            r"i\s+never\s+got)\b|(?:我没收到|我未收到|我没有收到|还没收到)",
            message, re.I,
        )
        if positive_receipt or negative_receipt:
            expected_received = not bool(negative_receipt)
            receipt_claims = [claim for claim in facts if claim.get("field") in {"received", "arrival_status", "shipping_status", "delivery_status"}]
            if not receipt_claims:
                reasons.append("undeclared_factual_claim:received")
            elif not any(
                (str(claim.get("value")).casefold() in {"true", "received", "arrived", "signed", "delivered"}) == expected_received
                for claim in receipt_claims
            ):
                reasons.append("factual_claim_metadata_mismatch:received")
    return list(dict.fromkeys(reasons))


def _fact_grounding_reasons(case: CaseSpec, simulation: Any, messages: list[str], tags: set[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Check a small set of concrete case facts without treating preferences as facts."""
    reasons: list[str] = []
    authorized: list[dict[str, str]] = []
    knowledge = case.user_knowledge or {}
    product_values = [
        str(value) for key, value in knowledge.items()
        if key in {"product_name", "product", "purchased_product"} and isinstance(value, (str, int, float))
    ]
    for event in getattr(simulation, "backend_events", []) or []:
        result = event.get("result", {}) if isinstance(event, dict) else {}
        data = result.get("data", result) if isinstance(result, dict) else {}
        for path, value in _scalar_strings(data or {}):
            leaf = path.rsplit(".", 1)[-1].lower()
            if leaf in {"name", "product_name", "product"} and any(part in path.lower() for part in ("product", "item", "order")):
                product_values.append(value)

    def categories(text: str) -> set[str]:
        lowered = text.casefold()
        return {
            category for category, labels in _PRODUCT_CATEGORIES.items()
            if any(label.casefold() in lowered for label in labels)
        }

    def shipping_statuses(text: str) -> set[str]:
        lowered = text.casefold()
        found = set()
        for status in _SHIPPING_STATUS_CUES:
            if re.search(
                rf"(?:\b(?:status|state)\b|(?:物流|订单)?状态)\s*(?:is\s*)?[:：=]?\s*{re.escape(status.casefold())}\b",
                lowered,
            ):
                found.add(status)
        for status, cues in _SHIPPING_STATUS_CUES.items():
            for cue in cues:
                start = lowered.find(cue.casefold())
                if start < 0:
                    continue
                # A negated delivery claim is not a claim that the package
                # was delivered. Unshipped cues are explicitly negative and
                # are therefore handled separately above.
                if status == "Signed":
                    prefix = lowered[max(0, start - 20):start]
                    if re.search(r"\b(?:not|never|no|haven't|hasn't|didn't|don't)\b[^.!?]{0,16}$", prefix):
                        continue
                    if re.search(r"(?:没|未|没有|尚未)[^。！？.!?]{0,10}$", prefix):
                        continue
                found.add(status)
                break
        return found

    known_shipping_statuses: set[str] = set()
    belief_shipping_statuses: set[str] = set()
    for key in ("shipping_status", "delivery_status"):
        value = knowledge.get(key)
        if value is not None:
            known_shipping_statuses.update(shipping_statuses(str(value)))
            canonical = str(value).strip().casefold()
            if canonical in {status.casefold() for status in _SHIPPING_STATUS_CUES}:
                known_shipping_statuses.add(next(
                    status for status in _SHIPPING_STATUS_CUES
                    if status.casefold() == canonical
                ))
    for key in ("believes_shipping_status", "believes_delivery_status"):
        value = knowledge.get(key)
        if value is not None:
            belief_shipping_statuses.update(shipping_statuses(str(value)))
            canonical = str(value).strip().casefold()
            if canonical in {status.casefold() for status in _SHIPPING_STATUS_CUES}:
                belief_shipping_statuses.add(next(
                    status for status in _SHIPPING_STATUS_CUES
                    if status.casefold() == canonical
                ))
    publicly_observed_statuses: set[str] = set()
    for event in getattr(simulation, "backend_events", []) or []:
        result = event.get("result", {}) if isinstance(event, dict) else {}
        data = result.get("data", result) if isinstance(result, dict) else {}
        if isinstance(data, dict):
            for key in ("shipping_status", "delivery_status", "order_status"):
                if data.get(key) is not None:
                    publicly_observed_statuses.update(shipping_statuses(str(data[key])))
                    canonical = str(data[key]).strip().casefold()
                    if canonical in {status.casefold() for status in _SHIPPING_STATUS_CUES}:
                        publicly_observed_statuses.add(next(
                            status for status in _SHIPPING_STATUS_CUES
                            if status.casefold() == canonical
                        ))
    for turn in getattr(simulation, "turns", []) or []:
        publicly_observed_statuses.update(shipping_statuses(_turn_agent_message(turn)))
    # A recorded Customer belief is a legitimate source even when it differs
    # from backend truth. Truth mismatch is evidence for the service task, not
    # evidence that the Customer fabricated their belief.
    authorized_shipping_statuses = (
        known_shipping_statuses | publicly_observed_statuses | belief_shipping_statuses
    )

    known_categories = set().union(*(categories(value) for value in product_values)) if product_values else set()
    deceptive = "deceptive_claim" in tags
    belief_values = " ".join(
        str(value) for key, value in knowledge.items()
        if str(key).startswith("believes_") and value is not None
    )
    belief_categories = categories(belief_values)
    for message in messages:
        claimed_shipping_statuses = shipping_statuses(message)
        for status in claimed_shipping_statuses - authorized_shipping_statuses:
            if status in belief_shipping_statuses:
                authorized.append({
                    "field": "shipping_status",
                    "authorization": "customer_belief",
                })
            elif deceptive:
                authorized.append({"field": "shipping_status", "authorization": "deceptive_claim"})
            else:
                reasons.append("unsupported_fact_generation:shipping_status")

        message_categories = categories(message)
        purchase_context = bool(_PURCHASE_CLAIM_CONTEXT.search(message))
        mismatched_categories = (
            message_categories - known_categories
            if purchase_context and message_categories else set()
        )
        if mismatched_categories:
            if deceptive:
                authorized.append({"field": "product_name", "authorization": "deceptive_claim"})
            elif belief_categories & mismatched_categories:
                authorized.append({"field": "product_name", "authorization": "customer_belief"})
            else:
                reasons.append("unsupported_fact_generation:product_name")

        for field, pattern in _ATTRIBUTE_PATTERNS.items():
            matches = [next((value for value in match if value), "") for match in pattern.findall(message)]
            matches = [value.strip().casefold() for value in matches if value]
            if not matches or not _PURCHASE_CLAIM_CONTEXT.search(message):
                continue
            aliases = _ATTRIBUTE_KEYS[field]
            known_attribute_values = {
                str(value).strip().casefold()
                for key, value in knowledge.items()
                if key in aliases and value is not None and not isinstance(value, bool)
            }
            if known_attribute_values and set(matches).issubset(known_attribute_values):
                continue
            belief_values_for_field = {
                str(value).strip().casefold()
                for key, value in knowledge.items()
                if str(key).startswith("believes_")
                and any(alias in str(key) for alias in aliases)
                and value is not None
            }
            if set(matches).issubset(belief_values_for_field):
                authorized.append({"field": field, "authorization": "customer_belief"})
                continue
            if deceptive:
                authorized.append({"field": field, "authorization": "deceptive_claim"})
            else:
                reasons.append(f"unsupported_fact_generation:{field}")

    # Unknown IDs are never a permitted form of deception: they can create
    # fictitious records and poison case-level attribution.
    known_ids = _known_identifiers(case, simulation)
    identifier_patterns = _identifier_claim_patterns()
    for message in messages:
        for field, pattern in identifier_patterns.items():
            for value in pattern.findall(message):
                if value.casefold() not in known_ids[field]:
                    reasons.append(f"unsupported_fact_generation:{field}")

    # This explicit structured case supports distinguishing a permitted lie
    # from an accidental LLM fabrication without inferring intent from prose.
    received = knowledge.get("has_received", knowledge.get("received"))
    if received is True and any(re.search(r"(?:没收到|未收到|没有收到|never received|did not receive|haven't received)", message, re.I) for message in messages):
        if deceptive:
            authorized.append({"field": "received", "authorization": "deceptive_claim"})
        else:
            reasons.append("unsupported_fact_generation:received")
    return list(dict.fromkeys(reasons)), authorized


def _policy_compliance_reasons(simulation: Any, case: CaseSpec, messages: list[str], tags: set[str]) -> list[str]:
    transcript = "\n".join(messages).casefold()
    reasons: list[str] = []
    if "pressure" not in tags and any(marker.casefold() in transcript for marker in _PRESSURE_MARKERS):
        reasons.append("policy_violation:pressure_not_allowed")
    if "cooperative" in tags and any(marker.casefold() in transcript for marker in _HOSTILE_MARKERS):
        reasons.append("policy_violation:cooperative_tone_violated")
    if not ({"contradiction", "delayed_contradiction"} & tags) and any(marker.casefold() in transcript for marker in _CONTRADICTION_MARKERS):
        reasons.append("policy_violation:contradiction_not_allowed")
    if "delayed_contradiction" in tags and "contradiction" not in tags:
        events = getattr(simulation, "backend_events", []) or []
        for index, message in enumerate(messages):
            if not any(marker.casefold() in message.casefold() for marker in _CONTRADICTION_MARKERS):
                continue
            verified_before = any(
                int(event.get("turn_index", -1)) < index
                and event.get("event_type") in {"tool_call", "tool_query"}
                and "query" in str(event.get("name", "")).casefold()
                and bool((event.get("result") or {}).get("success"))
                for event in events if isinstance(event, dict)
            )
            if not verified_before:
                reasons.append("policy_violation:delayed_contradiction_too_early")
                break
    if "authority_challenge" not in tags and any(marker.casefold() in transcript for marker in _AUTHORITY_CHALLENGE_MARKERS):
        reasons.append("policy_violation:authority_challenge_not_allowed")
    if "escalation" not in tags and any(marker.casefold() in transcript for marker in _ESCALATION_MARKERS):
        reasons.append("policy_violation:escalation_not_allowed")

    opening_contract = get_customer_opening_contract(case)
    required_opening_ids = {str(item["value"]).casefold() for item in opening_contract if item.get("field") in _ID_KEYS}
    if messages and ({"withholding", "delayed_disclosure"} & tags):
        for field, values in _known_identifiers(case, simulation).items():
            for value in values - required_opening_ids:
                if value and value in messages[0].casefold():
                    reasons.append("policy_violation:optional_information_disclosed_early")
                    break

    if not ({"withholding", "delayed_disclosure"} & tags):
        reveal_flags = {
            "order_id": "reveal_order_id_on_request",
            "customer_id": "reveal_customer_id_on_request",
            "record_id": "reveal_record_id_on_request",
        }
        known_ids = _known_identifiers(case, simulation)
        turns = getattr(simulation, "turns", []) or []
        for index in range(1, min(len(messages), len(turns))):
            agent_request = _turn_agent_message(turns[index - 1]).casefold()
            customer_reply = messages[index].casefold()
            for field, values in known_ids.items():
                requested = any(token in agent_request for token in {
                    "order_id": ("订单号", "订单编号", "order id", "order_id"),
                    "customer_id": ("客户号", "客户编号", "customer id", "customer_id"),
                    "record_id": ("记录号", "记录编号", "record id", "record_id"),
                    "booking_id": ("订单号", "booking id", "ticket number"),
                    "ticket_id": ("票号", "机票号", "ticket id", "ticket number"),
                    "account_id": ("账户号", "account id"),
                }.get(field, ()))
                if not requested:
                    continue
                reveal_flag = reveal_flags.get(field)
                can_reveal = field in _ID_KEYS and bool(values) and (
                    reveal_flag is None or (case.user_policy or {}).get(reveal_flag, True)
                )
                if can_reveal and not any(value in customer_reply for value in values):
                    reasons.append(f"policy_violation:known_{field}_withheld_when_requested")
    return list(dict.fromkeys(reasons))


def _hidden_candidates(case: CaseSpec) -> list[tuple[str, str, bool]]:
    candidates: list[tuple[str, str, bool]] = []
    metadata_values = (case.metadata or {}).get("backend_system_variables", {})
    for path, text in _scalar_strings(metadata_values, "backend_system_variables"):
        candidates.append((path.rsplit(".", 1)[-1], text, False))
    private_variables = ((case.backend_record or {}).get("private_state") or {}).get("system_variables", {})
    for path, text in _scalar_strings(private_variables, "private_state.system_variables"):
        candidates.append((path.rsplit(".", 1)[-1], text, False))

    # Collect authoritative identifiers from the record, but not arbitrary
    # business text that may include public labels or ordinary vocabulary.
    for path, text in _scalar_strings(case.backend_record or {}):
        leaf = path.rsplit(".", 1)[-1].split("[", 1)[0]
        normalized = re.sub(r"[^a-z0-9]+", "_", leaf.lower()).strip("_")
        if normalized in _ID_KEYS or _IDENTIFIER_KEY.search(normalized):
            candidates.append((leaf, text, True))

    deduped = {}
    for field_name, value, is_identifier in candidates:
        if value and value.lower() not in {"none", "null", "unknown"}:
            deduped[(field_name, value.lower())] = (field_name, value, is_identifier)
    return list(deduped.values())


def _value_mentioned(message: str, value: str, identifier: bool) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    # Very short enum values are too ambiguous for a deterministic exact-text
    # leakage test. Identifiers are checked even when short.
    if not identifier and len(normalized) < 3:
        return False
    if identifier or re.search(r"[0-9_-]", normalized):
        return normalized.casefold() in message.casefold()
    return re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", message, re.IGNORECASE) is not None


def assess_customer_behavior(simulation: Any, customer_policy: Any) -> CustomerBehaviorAssessment:
    """Assess hard Customer-side contracts from a completed simulation trace."""
    reasons: list[str] = []
    checks: dict[str, Any] = {}
    case = _case_spec(getattr(simulation, "case_spec", None))
    turns = getattr(simulation, "turns", []) or []
    messages = [str(getattr(turn, "user_message", "") or "") for turn in turns]
    nonempty = bool(messages) and all(message.strip() for message in messages)
    checks["customer_messages_nonempty"] = nonempty
    if not messages and (case is not None or simulation.__class__.__name__ == "SimulationResult"):
        reasons.append("customer_messages_missing")
    elif messages and not nonempty:
        reasons.append("customer_message_empty")

    if case is None:
        checks["case_spec_available"] = False
    else:
        checks["case_spec_available"] = True
        try:
            CustomerPolicyCompiler().compile(customer_policy, case)
            checks["customer_policy_immutable_case"] = True
        except Exception:
            checks["customer_policy_immutable_case"] = False
            reasons.append("customer_policy_invalid")

        opening_contract = get_customer_opening_contract(case)
        opening_ok = True
        if opening_contract:
            opening = messages[0] if messages else ""
            opening_ok = all(str(item["value"]).casefold() in opening.casefold() for item in opening_contract)
        checks["mandatory_opening_disclosure"] = opening_ok
        if not opening_ok:
            reasons.append("mandatory_opening_disclosure_missing")

        environment_state = getattr(simulation, "user_environment_state", {}) or {}
        if hasattr(environment_state, "to_dict"):
            environment_state = environment_state.to_dict()
        observed_goal = environment_state.get("goal") if isinstance(environment_state, dict) else None
        goal_unchanged = observed_goal is None or observed_goal == case.user_goal
        checks["case_goal_unchanged"] = goal_unchanged
        if not goal_unchanged:
            reasons.append("customer_goal_mutated")

        transcript = "\n".join(messages)
        if _INTERNAL_FIELD.search(transcript):
            reasons.append("customer_evaluator_field_leakage")
        allowed = _known_values(case, simulation)
        leaked_fields = set()
        unknown_identifiers = set()
        for field_name, value, is_identifier in _hidden_candidates(case):
            if value.lower() in allowed or not any(_value_mentioned(message, value, is_identifier) for message in messages):
                continue
            if is_identifier:
                unknown_identifiers.add(field_name)
            else:
                leaked_fields.add(field_name)
        for field_name in sorted(leaked_fields):
            reasons.append(f"customer_hidden_value_leakage:{field_name}")
        for field_name in sorted(unknown_identifiers):
            reasons.append(f"customer_unknown_identifier_leakage:{field_name}")

        tags = _policy_tags(customer_policy)
        policy_reasons = _policy_compliance_reasons(simulation, case, messages, tags)
        checks["policy_compliance"] = {
            "valid": not policy_reasons,
            "strategy_tags": sorted(tags),
            "reasons": policy_reasons,
        }
        reasons.extend(policy_reasons)

        tags = _policy_tags(customer_policy)
        fact_reasons, authorized_claims = _fact_grounding_reasons(
            case, simulation, messages, tags
        )
        claim_reasons, claim_validation, structured_authorizations, declared_by_turn = (
            _structured_claim_validation(case, simulation, tags)
        )
        undeclared_reasons = _undeclared_concrete_claims(messages, declared_by_turn)
        fact_reasons.extend(claim_reasons)
        fact_reasons.extend(undeclared_reasons)
        authorized_claims.extend(structured_authorizations)
        claim_metadata_records = []
        for turn_index, turn in enumerate(turns):
            metadata = _turn_claim_metadata(turn)
            if metadata:
                claim_metadata_records.append({
                    "turn_index": turn_index,
                    "parse_status": metadata.get("parse_status"),
                    "parse_error": metadata.get("parse_error"),
                    "claims": metadata.get("declared_claims", metadata.get("claims", [])),
                    "raw_envelope": metadata.get("raw_envelope"),
                })
        checks["factual_grounding"] = {
            "valid": not fact_reasons,
            "reasons": list(dict.fromkeys(fact_reasons)),
            "policy_authorized_claims": authorized_claims,
            "claim_validation": claim_validation,
            "undeclared_claim_reasons": undeclared_reasons,
            "customer_claims_declared": claim_metadata_records,
            "customer_claims_validation": claim_validation,
            "factual_claim_metadata_parse_status": [record["parse_status"] for record in claim_metadata_records],
            "factual_claim_metadata_parse_error": [record["parse_error"] for record in claim_metadata_records],
        }
        if "deceptive_claim" in tags:
            checks["deception_authorization"] = "explicit_policy_tag"
        elif "mistaken_belief" in tags:
            checks["deception_authorization"] = "recorded_belief_strategy; belief_expression_does_not_require_tag"
        else:
            checks["deception_authorization"] = "deception_not_authorized; recorded_beliefs_remain_valid"
        reasons.extend(fact_reasons)

    role_drift = any(_SERVICE_VOICE.search(message) for message in messages)
    checks["customer_not_speaking_as_service"] = not role_drift
    if role_drift:
        reasons.append("customer_role_drift")
    policy_text = json.dumps(
        customer_policy.to_dict() if hasattr(customer_policy, "to_dict") else customer_policy,
        ensure_ascii=False,
    )
    checks["role_drift_suspected"] = bool(_SERVICE_VOICE.search(policy_text))
    reasons = list(dict.fromkeys(reasons))
    return CustomerBehaviorAssessment(
        valid=not reasons,
        hard_violation=bool(reasons),
        reasons=reasons,
        checks=checks,
    )
