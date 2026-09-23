# Seed 7 原始轨迹有效性审计：空订单查询与空首轮用户消息

- 审计范围：Seed 7 的 Static、Customer-only、Coevolution REAL 轨迹；按 (mode, simulation_id) 去重。
- 原始轨迹归档提交：77e3ae8045788776de7297d5c217828e25b33ad8
- Runtime freeze：1fc5aadf4e9294ef14d59f8ca096943b6d38eb67（exp-freeze-2026-09-22-calibrated）
- 机器可读审计：[analysis/seed07_raw_trajectory_audit_20260923.json](../analysis/seed07_raw_trajectory_audit_20260923.json)
- 只读范围：没有调用模型/evaluator，没有重跑 Seed 7，没有改 prompt、schema、runtime、正式配置或已完成结果。

## 结论先行

1. query_order({"order_id": ""}) 不只在 adaptive Customer 下出现：Static 的 3 个缺少订单号开场中 3/3 都空参数查询并被 backend 拒绝；有确切 ID 的配对轨迹 3/3 都用非空 ID 成功查单。
2. 全部 86 个空参数查询中，Agent reasoning 都明确提到订单号缺失或应该先询问，但同一轮仍发出空参数工具调用；86 条都在 turn 0，均收到 order_not_found 并以 tool_failure 结束。
3. 根因最符合 Agent 决策与指令优先级/宽松工具契约的交互，而不是单次随机失误。Adaptive Customer 可增加缺 ID 输入，但不是缺陷的起源。
4. 另有 70/204 个 episode 的第一条用户消息是空字符串。CaseSpec 都有用户已知订单号且 show_order_id_initially=true。47 条被记录为 valid，43 条是 task failure；39 条 valid failure 进入 Customer candidate fitness，7 条进入 Service source-failure set。这是 POTENTIAL FORMAL VALIDITY BLOCKER，与 Agent 空查询分开。

## Empty-order 查询频率

初始消息中的订单号按该 simulation 对应 CaseSpec 的确切 ID 匹配，不是搜索任意 ORD-。Ask-user-first 只统计首次 backend tool call 之前的更早一轮明确询问订单号；同一轮既发空 query 又在 chat 问 ID，不算先问后查。

| Mode | Episodes | 空首条消息 | 初始无确切 ID | 初始有确切 ID | 首工具 query_order | 空参数 | 非空参数 | 缺参数键 | 没有 query_order | Ask-user-first | tool_failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static | 6 | 0 | 3 | 3 | 6 | 3 | 3 | 0 | 0 | 0 | 3 |
| Customer-only | 87 | 31 | 71 | 16 | 85 | 34 | 51 | 0 | 2 | 32 | 37 |
| Coevolution | 111 | 39 | 85 | 26 | 110 | 49 | 59 | 2 | 1 | 33 | 51 |

| Mode | P(empty query | 初始无 ID) | P(empty query | 初始有 ID) | order_not_found |
|---|---:|---:|---:|
| Static | 3/3 = 100% | 0/3 | 3 |
| Customer-only | 34/71 = 47.9% | 0/16 | 37 |
| Coevolution | 49/85 = 57.6% | 0/26 | 51 |

按 cache key 恢复 generation 后的频率：

| Mode / Gen | N | 空首条消息 | 无 ID / 有 ID | 首工具 query_order | 空 / 非空 / 缺 key | Ask-user-first | tool_failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static G0 | 2 | 0 | 1 / 1 | 2 | 1 / 1 / 0 | 0 | 1 |
| Static G1 | 2 | 0 | 1 / 1 | 2 | 1 / 1 / 0 | 0 | 1 |
| Static G2 | 2 | 0 | 1 / 1 | 2 | 1 / 1 / 0 | 0 | 1 |
| Customer-only G0 | 26 | 11 | 19 / 7 | 25 | 9 / 16 / 0 | 9 | 9 |
| Customer-only G1 | 31 | 7 | 24 / 7 | 31 | 13 / 18 / 0 | 8 | 16 |
| Customer-only G2 | 30 | 13 | 28 / 2 | 29 | 12 / 17 / 0 | 15 | 12 |
| Coevolution G0 | 32 | 11 | 21 / 11 | 32 | 13 / 19 / 0 | 8 | 13 |
| Coevolution G1 | 39 | 9 | 30 / 9 | 39 | 20 / 18 / 1 | 9 | 21 |
| Coevolution G2 | 40 | 19 | 34 / 6 | 39 | 16 / 22 / 1 | 16 | 17 |

所有空 query 的逐条 ID 在机器文件的 empty_query_evidence.simulation_ids；完整 user message、reasoning、arguments 和 backend 结果可在 [Seed 7 原始 episodes](seed07_raw_artifacts/episodes.jsonl) 按 simulation_id 读取。86 条 reasoning 逐一确认缺少 ID；代表性原文：

> “The user hasn't provided an order number. The environment says we must query the order first. Since we don't have order_id, we should ask in chat.”

有些 episode 先前一轮问过 ID、用户补充后才查工具；这些计入 Ask-user-first，不属于空参数错误。86 个空参数 case 全部在首轮，没有先问后查；若同一轮 chat 同时问 ID，它仍然已经先发出了空参数 tool call。

## Static 六条逐例审计

| Gen | Intent | 首条消息含确切 ID | First tool / arguments | Backend result | 后续工具 | Termination |
|---|---|---|---|---|---|---|
| G0 | user_return_high_credit | 否 | query_order({"order_id": ""}) | order_not_found | 无 | tool_failure |
| G0 | user_return_low_credit_no_doc | 是 | query_order({"order_id": "ORD-29DBFF5311"}) | 成功，shipping_status=Signed | query_customer_profile, request_document | user_ended |
| G1 | user_return_high_credit | 否 | query_order({"order_id": ""}) | order_not_found | 无 | tool_failure |
| G1 | user_return_low_credit_no_doc | 是 | query_order({"order_id": "ORD-6959C9BD34"}) | 成功，shipping_status=Signed | request_document | no_more_user_messages |
| G2 | user_return_high_credit | 否 | query_order({"order_id": ""}) | order_not_found | 无 | tool_failure |
| G2 | user_return_low_credit_no_doc | 是 | query_order({"order_id": "ORD-3673D0006C"}) | 成功，shipping_status=Signed | request_document | user_ended |

三代配对模式与 artifacts 一致。非空 ID 查询成功并进入后续工具流程，但这三条都没有达到 goal_solved。

## Prompt / tool contract

- [ecommerce_refund_prompts.py](../framework/prompts/ecommerce_refund_prompts.py#L61)：一边要求先用 query_order 查询，另一边要求订单号不足时先在 chat 询问。这不是严格逻辑矛盾，但“先查询”的无条件措辞，加上运行时追加的“后台状态不可猜测，必须先调用查询工具”，产生了可观测的步骤优先级歧义。
- [ecommerce.py](../framework/backend/ecommerce.py#L14)：order_id 是 required string，没有 minLength: 1。
- [base.py](../framework/backend/base.py#L204)：schema builder 没有非空约束。
- [agent_model.py](../framework/models/agent_model.py#L1191)：参数解析后直接调用 execute_tool，执行边界没有 JSON Schema validation。arguments_valid=true 只表示 JSON 可解析。
- [ecommerce.py](../framework/backend/ecommerce.py#L77)：backend 对空 ID 正确返回 order_not_found。

因此空字符串符合当前 string schema 且没有运行时拦截，能够到达 backend。

## 空首轮 Customer message：有效性风险

共 70/204 = 34.3%，Customer-only 31 条（G0/G1/G2：11/7/13），Coevolution 39 条（11/9/19）；Static 0 条。之后出现非空 Customer 消息的 54 条（25 + 29），另 16 条没有后续非空消息。

| Mode | 空首条消息 | Valid / invalid-only | Valid task failure / success | Termination counts |
|---|---:|---:|---:|---|
| Customer-only | 31 | 18 / 13 | 17 / 1 | end_step 12, user_ended 6, max_turns 5, tool_failure 5, no_more_user_messages 2, goal_fulfilled 1 |
| Coevolution | 39 | 29 / 10 | 26 / 3 | tool_failure 9, user_ended 9, end_step 9, max_turns 5, no_more_user_messages 4, goal_fulfilled 3 |
| Total | 70 | 47 / 23 | 43 / 4 |  |

所有 204 条 CaseSpec 都标明 Customer 知道订单号且 show_order_id_initially=true。订单号被放入 Customer 已知信息；首轮提示还要求不要为了多轮隐藏必要事实。LLM user 的 generate_initial_message() 对响应做 strip/清理后直接返回；空内容不会 fallback。原始 User provider completion 未保存，无法区分 provider 原始空文本与空白文本经 strip 后变空。

影响进化数据的范围：

- Customer candidate/elite score cohorts 内有 66 个空首轮记录：43 valid（39 failures、4 successes），23 invalid。39 个 valid failure 被当前 selector 当作 legitimate attack signal 计入 Customer fitness。
- 选中 policy 的评分 cohort 内有 18 个空首轮记录：13 valid failures、5 invalid。
- Service source failures 内有 7 个 valid 空首轮失败：G0 2、G1 1、G2 4。
- generation 输出中另有一条 Coevolution G2 validation 的空首轮记录，valid 且 task_success=false。

CustomerSelector 排除 protocol-invalid，但不检查空 Customer message；valid 空消息 failure 会进入 attack-success 聚合。coverage 还对 invalid episode 的 sop_node 计数。本审计没有重新评分，只核对存储状态和 selector/evaluator 语义。

> POTENTIAL FORMAL VALIDITY BLOCKER：空 Customer 首轮不是已证实的合法对抗策略，却作为 valid 业务失败进入 Customer fitness 和 Service source failures；它混淆 Customer evolution 与 service robustness 的归因。

## Customer candidate 语义角色

人工分类，不调用 LLM judge。完整字段保留在 [evolution_outputs.jsonl](seed07_raw_artifacts/evolution_outputs.jsonl)；机器文件记录 18 个候选的 name、tags、分类。

| Mode / Gen | Candidate | Manual class |
|---|---|---|
| Customer-only G0 | Authoritative Verification Before Comfort | SERVICE_BEHAVIOR |
|  | Direct Correction of Claimed Success | SERVICE_BEHAVIOR |
|  | Escalation After Repeated Incomplete Execution | MIXED |
| Customer-only G1 | Required Identifier Clarification Before Action | SERVICE_BEHAVIOR |
|  | Delayed Disclosure of Unverified Failure | SERVICE_BEHAVIOR |
|  | Escalation on Persistent Verification Failure | MIXED |
| Customer-only G2 | Required Identifier Gate with Delayed Contradiction | MIXED |
|  | Authority-Challenging Completion Demand | USER_BEHAVIOR |
|  | Cooperative Paraphrase Re-request | MIXED |
| Coevolution G0 | Confirmed_Only_Action | SERVICE_BEHAVIOR |
|  | Delayed_Disclosure_Recovery | MIXED |
|  | Escalation_After_Failed_Action | MIXED |
| Coevolution G1 | Persistent Verification Advocate | USER_BEHAVIOR |
|  | Escalation-Driven Authority Challenger | USER_BEHAVIOR |
|  | Gradual Disclosure Paraphraser | USER_BEHAVIOR |
| Coevolution G2 | Verification-First Persistence | MIXED |
|  | Delayed-Claim Contradiction | USER_BEHAVIOR |
|  | Escalating Authority Challenge | USER_BEHAVIOR |

共 SERVICE_BEHAVIOR 5、MIXED 6、USER_BEHAVIOR 7。指定的前两个候选有明显 service-role drift；Escalation After Repeated Incomplete Execution 混合了用户要求升级和服务端升级动作。

CustomerPolicy.runtime_guidance() 会附加进 LLM Customer 的 system prompt。实际输出仍是客户话术，没有证据显示模型直接扮演客服；但没有字段级 counterfactual，不能因果断言具体 policy 字段造成了哪条用户行为。Customer-only G0/G1 选中的策略包含服务端规则，表明这些规则实际进入过运行提示。

## Service causal-target audit

9 个 patch 的 evaluation 都 valid、delta=0，并以 latest_attack_filter:insufficient_adversarial_improvement 被拒。

| Gen | Category | Patch 目标 | 是否明确“缺 required argument 时先问再调用” |
|---|---|---|---|
| G0 | VERIFICATION | final decision 前核验后台事实 | 否 |
| G0 | TOOL_SELECTION | 选择必要工具、避免重复调用 | 否 |
| G0 | ACTION_GROUNDING | 工具成功后才宣称动作完成 | 否 |
| G1 | VERIFICATION | final 前核验状态/动作结果 | 否 |
| G1 | ACTION_GROUNDING | 执行确认后报告并对齐目标 | 否 |
| G1 | RECOVERY | 工具失败后 retry/switch/escalate | 否 |
| G2 | VERIFICATION | final 前确认承诺动作发生 | 否 |
| G2 | TOOL_SELECTION | 按信息需要选工具，失败后 retry/解释 | 否 |
| G2 | ACTION_GROUNDING | final action 对齐目标和验证状态 | 否 |

**CAUSAL PRECONDITION TARGET MISSED。** 没有规则要求工具调用前确认必填 identifier 已知且非空；不少规则关注工具失败后的恢复或最终话术，时间顺序已晚于空参数调用。

## 正式结果含义与预声明 ablation

- 空 query 是 systematic baseline weakness + prompt/tool-contract interaction，不是 isolated stochastic error。
- Static 三代都稳定复现。Adaptive Customer 应描述为暴露/放大 latent weakness，而不是创造缺陷。
- 保留正式 run 原貌。Static Agent 弱点仍有直接证据；Customer fitness 和 Service source attribution 必须披露空首轮问题，避免把模拟器空输出都解释为合法 attack。
- [pilot_analysis_plan.md](pilot_analysis_plan.md#L59) 已预声明两个独立 ablation：manual missing-ID ServicePolicy；query_order.order_id minLength:1，并要求使用同 CaseSpec/CustomerPolicy/model/seed-config。计划文件没有明确预注册 combined arm；如要做 combined，应先作为独立扩展登记。
- 当前没有 execution-boundary JSON Schema validator，所以 minLength 是 provider-exposed 限制，不是 backend 硬拦截。此次未跑 ablation。
- Generation 是 trajectory axis，不是独立样本。此次仅审查 Seed 7 事实，不作统计显著性结论。
