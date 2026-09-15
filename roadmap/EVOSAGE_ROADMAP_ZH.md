# EvoSAGE Roadmap

更新时间：2026-09-15

本文记录当前对 SAGE-Bench 的观察、EvoSAGE 的目标定义，以及后续实施计划。

## 1. 当前状态

### 已完成

- 已将项目改造成纯 OpenAI-compatible API 模式。
- User、Agent、Judge 可以分别通过同一个 API 接口调用。
- 已使用 `dashscope/qwen3.7-plus` 完成过端到端 API 冒烟测试。
- API Key 已通过 macOS Keychain 持久化保存。
- 已创建并推送到私有 GitHub 仓库 `hjy8848/EvoSAGE`。
- 已修复 API 模型名包含 `/` 时结果文件名异常的问题。
- 本次评测过程中发现 `run.sh` 的日志文件名也需要对 `/` 做安全化处理，已在本地修复。

### 已验证的单轮样例

样例场景：`online_education / seek_answer`

用户问题是询问 Python 函数默认参数的实际用法。模型输出了：

> 您好！很高兴为您解答默认参数的应用，稍后为您发送详细指导。

结论：

- API 调用成功；
- JSON 解析成功；
- 分类、路径和最终动作字段均成功生成；
- 但模型没有真正解释问题，只是承诺稍后提供指导。

因此，单轮测试证明的是“评测管线可以运行”，不是“用户问题已经解决”。

### 已尝试的完整评测

配置：

- 场景：`online_education`；
- 5 个用户 Intent；
- 每个 Intent 2 个用户，共 10 个样本；
- 最大对话轮数 10；
- User、Agent、Judge 均使用 `dashscope/qwen3.7-plus`；
- 串行运行。

结果：

- 前 2 个 `seek_answer` 样本完成，均得到 `0.8592`；
- 第 2 个样本经历了一次 300 秒 API 超时后重试成功；
- Judge 曾发现“重复反馈”标记与规则路径不一致，并触发重试；
- 评测在第 3 个样本期间暂停；
- 因为程序在全部样本结束后才统一保存结果，所以这 2 个样本的具体对话没有落盘。

这说明当前完整评测的主要瓶颈是 API 延迟、超时重试和评测结果的增量保存，而不仅是模型得分。

### 2026-09-15：并发 8 的最小多轮冒烟测试

配置：

- 场景：`online_education / seek_answer`；
- 样本数：2；
- 最大对话轮数：5；
- User、Agent、Judge：`dashscope/qwen3.7-plus`；
- 并发参数：`--max-workers 8`；
- 评测策略：`intent_based`。

结果：

- 2 个样本均完成，10 轮对话均成功落盘；
- JSON 解析错误率：`0`；
- 平均总体得分：`0.7672`；
- 分类准确率：`0.7083`；路径正确率：`0.8571`；最终动作正确率：`0.6000`；话术质量：`0.8513`；
- 但两个样本都没有稳定地在当前回复中解释 Python 默认参数。典型回复是“已为您安排导师解答，请稍候”，而不是解释、推导或示例。

这次结果确认了一个比模型分数更优先的问题：当前 Agent 的行为契约是“客服 SOP 路由”，不是“在线教学问答”。

## 2. 对当前 Bench 的关键观察

### 2.1 当前用户模拟器不是完整业务环境

当前用户模拟器主要由以下部分组成：

```text
固定 Intent 和背景
+ 对抗强度
+ LLM 生成第一句和后续消息
+ 最近几轮对话上下文
+ 基于 Agent 动作名称的满意度加减
+ 基于关键词的问题解决判断
+ 基于关键词和重复文本的终止判断
```

它还没有明确维护以下隐藏业务状态：

- 用户真实目标；
- 已知和未知的业务事实；
- 已向客服透露的信息；
- 是否满足业务条件；
- 每个客服动作对业务状态的实际影响；
- 用户目标是否真正达成。

因此，当前模拟器更接近“会继续聊天的 LLM 用户”，而不是“可执行的业务环境”。

### 2.2 当前得分不等于用户目标完成

当前评分中，分类、路径和最终动作占主要比重。模型只要输出结构正确的：

```json
{"Action": "PLAN", "PLAN": "PLAN_B"}
```

就可能获得较高逻辑分，即使自然语言回复没有真正回答用户问题。

需要新增独立的用户目标完成指标，例如：

- 是否直接回答问题；
- 是否提供必要解释或步骤；
- 是否提供必要示例；
- 是否收集完成业务动作所需的信息；
- 用户目标是否达到；
- 是否应该继续对话而不是过早结束。

### 2.3 多轮结束条件存在语义问题

之前的单轮样例之所以只运行一轮，是因为测试参数设置了 `max_turns=1`。这代表程序正常结束，不代表用户满意或问题解决。

完整评测还需要区分：

```text
达到最大轮次
Agent 主动结束
用户目标完成
用户主动离开
重复对话失败
API 或解析错误
```

这些状态不能统一记录为 `completed`。

### 2.4 路径应区分预测、执行和标准路径

当前输出中存在 `expected_path` 和 `path_taken`，但两者的语义仍不够清晰。需要明确记录：

```text
predicted_path：模型预测的 SOP 路径
executed_path：环境根据实际动作执行的路径
gold_path：规则引擎给出的标准路径
```

否则模型可能只是在 JSON 中声称自己走了正确路径，而不是实际执行了正确决策。

### 2.5 User、Agent、Judge 使用同一模型存在偏差

纯 API 模式允许三个角色使用不同模型，但当前测试使用同一个模型作为 User、Agent 和 Judge。这会带来：

- 语言风格相关性；
- Judge 对 Agent 风格的偏好；
- 用户模拟器和客服模型共享相似的错误模式；
- 评测结果对单一模型分布过度敏感。

后续必须加入跨模型、跨 Prompt 和规则用户模拟器测试。

### 2.6 Agent 为什么不直接解答用户问题

2026-09-15 的 `seek_answer` 冒烟样本显示，Agent 不是缺少 Python 知识，而是在遵守当前 Prompt 和输出接口：

1. Agent 系统 Prompt 的首要任务是执行 SOP、判断分类、输出路径和最终动作；`PLAN_A/B/C` 被定义为资源分配方案，没有要求知识问题必须在当前轮直接解决。
2. `chat` 被限制在 40 字以内，并要求特殊内容不要直接包含代码，导致“解释默认参数并给出示例”与输出约束发生冲突。
3. 评测使用 `use_llm_for_full_output=True`，让同一次 LLM 调用同时生成分类、路径、动作和聊天文本。模型优先保证 JSON 和流程字段稳定，回复容易退化成“安排导师”“请稍候”等安全模板。
4. `AgentModel._process_turn_with_llm` 实际发送的消息只有 Agent system prompt 和对话历史；调用方传入的 `context_data`（包括 `system_info`）没有拼进 LLM 请求。代码中的“将收到系统信息”与实际输入不一致。
5. 当前 `PLAN` 回复没有连接到知识库、课程内容或教学工具，因此“分配资源”只是一个标签，不会自动产生真实答案。
6. 评测主要检查分类、路径、最终动作和话术礼貌程度，没有强制检查答案是否包含问题所需的解释、步骤和示例。LLM 用户模型还会根据“已处理”“祝您学习顺利”等关键词把问题标记为解决，使“承诺稍后回答”也可能得到较高分。

因此，后续不能把这类得分解释为“Agent 已经会答题”。在修复前，`seek_answer` 的核心验收标准应是：当前回复直接回答问题；必要时给出可验证的解释或示例；如果还需要人工支持，也必须先回答当前可回答的部分。

### 2.7 当前已落地的闭环 MVP（2026-09-15）

电商退款场景已经从“隐藏变量 + 规则判分”迁移为最小可执行闭环：

```text
CaseSpec
  ↓
EcommerceBackend（权威订单/客户状态）
  ↓
Customer Simulator（只知道用户自己的事实）
  ↕
Agent + query_order/query_customer_profile/query_payment
  ↓
Action Execution
  ↓
订单状态转移与 BackendEvent
  ↓
Evaluator（工具核验 + 状态结果 + 对话指标）
```

已完成的代码能力包括：

- `CaseSpec`：固定案例、用户目标、用户已知事实、用户披露策略、期望后端结果；
- `EcommerceBackend`：订单、客户、支付状态，以及退款、拦截、取件、换货、补材料、拒绝等动作；
- OpenAI-compatible `tool_calls`：支持原生工具调用，也兼容模型返回 JSON 工具调用；
- 工具和动作审计日志：记录调用参数、公开结果、状态转移和轮次；
- Customer Simulator 只接收公开业务事件，不接收完整隐藏 `backend_record`；
- `goal_solved` 由后端期望状态决定，不再只由“已解决”等关键词决定；
- 新增 `backend_verification` 指标，检查是否用正确订单号查询并完成目标状态转移；
- 旧版 PathList 不含 `action_*` 节点的问题已在路径评分中兼容；
- 现有回归测试覆盖后端查询、动作前置核验、状态转移和 Agent 工具循环；本轮新增 8 个 Ecommerce E2E 验收测试，并增加从 `LLMEvaluationPipeline.run_single_simulation` 主入口捕获 runner 接线的集成测试。

本轮验收范围锁定为 `ecommerce_refund`：runner 只为该场景创建并传递 `EcommerceBackend`，其余五个场景不进入新的 environment evaluation。仓库中保留的通用适配器不视为本轮完成项，真实 API 的电商退款 E2E trace 仍是下一步验收工作。

本轮环境修复的边界已经锁定：不实现自进化，不修改官方 SOP 图、隐藏测试集或评分标准；新增 `legacy_score` 与 `environment_score` 双轨输出，默认只允许正式动作工具执行，旧版 `finals.Action` 直执行必须显式开启 `--legacy-execution`。

本轮主链已按以下方式落地：

```text
User message
  ↓
Agent + 正式 tools
  ↓
query_order / query_customer_profile / query_payment
  ↓
EcommerceBackend tool result
  ↓
Agent tool loop（最多 8 步）
  ↓
submit_refund 等 action tool
  ↓
同一个 EcommerceBackend 的状态转移和 BackendEvent
  ↓
Simulator 记录事件与最终状态
  ↓
Evaluator 计算 legacy/environment 双轨结果
```

Agent 只看到 `system_prompt`、对话历史、`initial_observation` 和公开工具结果；完整 `backend_record`、`expected_outcome`、GT path/action 和评测 metadata 只保留给 Simulator/Evaluator。`predicted_path`/`predicted_action` 来自 Agent 输出，`executed_path`/`executed_action` 由 Backend event log 重建。真实目标完成只由 `expected_outcome` 与 Backend 最终状态判断，关键词结果仅保留在 legacy `goal_fulfillment` 中。

本轮新增的 Ecommerce E2E 覆盖：隐藏状态防泄漏、不同 Backend 的初始输入一致、用户错误陈述、查询结果误判、假装退款成功、真实退款成功、拒绝模型自报 executed_path、同一 Backend 的完整 trace，以及 runner 主入口接线。当前本地 `unittest discover -s tests` 结果为 15/15 通过。

## 3. 我们真正要进化什么

EvoSAGE 的进化目标应定义为：

> 在不同用户表达和业务状态下，稳定执行正确 SOP 的 operational policy，而不是拟合某一个用户模拟器的语言习惯。

进化对象按优先级排序如下。

### 第一优先级：状态识别策略

从用户语言和系统信息中正确识别：

- 问题是否清晰；
- 是否与业务对象相关；
- 是否重复反馈；
- 情绪状态；
- 资源依赖度；
- 是否触发退款或升级。

经验应保存为抽象业务条件，而不是原始用户句子。

### 第二优先级：SOP 节点转移策略

Agent 应该学会在状态条件下选择下一个 SOP 节点，而不是只生成一条看似正确的路径字符串。

### 第三优先级：最终动作策略

进化应重点提升以下映射的稳定性：

```text
业务状态 → 最终 Action → Action 参数
```

例如：

```text
重复反馈 → REVIEW
明确不满 → COMFORT
普通用户且满足退款条件 → REFUND
风险用户 → NEGOTIATE
```

### 第四优先级：节点级业务技能

为每个 SOP 节点形成可复用技能，例如：

- 情绪识别技能；
- 重复投诉识别技能；
- 退款资格核验技能；
- 必要信息追问技能；
- 安抚后继续业务处理技能。

### 第五优先级：交互策略和话术

最后再优化：

- 追问方式；
- 信息披露顺序；
- 安抚方式；
- 结束时机；
- 解释和示例质量。

话术分提升不能替代逻辑、动作和目标完成度提升。

## 4. 目标架构

### 4.1 固定层

以下内容在标准评测轨道中保持冻结：

- 官方 SOP 图；
- 标准路径；
- 分类字段定义；
- 最终动作定义；
- 评分公式；
- 隐藏测试集；
- 基本 Judge 标准。

### 4.2 可进化层

以下内容允许版本化进化：

- 状态识别 Prompt 或策略；
- SOP 节点级技能；
- 结构化失败记忆；
- 追问策略；
- 动作选择策略；
- 话术模板和沟通策略。

### 4.3 业务环境层

用户模拟器应升级为：

```text
隐藏业务状态
    ↓
用户行为策略
    ↓
LLM 自然语言实现
```

示例隐藏状态：

```json
{
  "goal": "获得退款",
  "facts": {
    "course_content_missing": true,
    "repeated_complaint": true,
    "is_risk_user": false
  },
  "emotion": "dissatisfied",
  "revealed_facts": [],
  "resolution_status": "unsolved"
}
```

Agent 不能直接读取完整状态，只能通过对话获取信息。Agent 的动作应改变环境状态，环境再生成可验证的结果反馈。

## 5. 分阶段实施计划

### Phase 0：修复评测可信度

目标：先确保分数代表真实能力。

- 增加增量保存：每个样本完成后立即写入 JSONL；
- 记录 `termination_reason` 的细分值；
- 分离 `predicted_path`、`executed_path` 和 `gold_path`；
- 增加 `goal_fulfillment` 指标；
- 增加回答内容检查，避免“稍后为您指导”被视为已回答；
- 修正 Agent 输入契约：将 `context_data/system_info` 实际注入 LLM 请求，并记录最终发送给模型的 Prompt 版本；
- 将“流程决策”和“用户答案”解耦：至少为 `seek_answer` 增加答案字段或答案完成标记，禁止 PLAN 回复替代知识解答；
- 放宽教学场景的回复长度和代码示例约束，增加“必须当场回答”的场景级 Prompt；
- 修复 Judge 与规则引擎之间的冲突记录；
- 增加每个请求的耗时、超时和重试次数；
- 为 User、Agent、Judge 分别记录模型和 Prompt 版本。

验收标准：中途停止后仍能查看已经完成的每个样本。

### Phase 1：重构用户模拟器

目标：从“LLM 话术生成器”升级为“可执行业务环境”。

- 定义统一的 `UserEnvironmentState`；
- 保留用户目标、业务事实、已披露事实、情绪和解决状态；
- 定义 Agent 动作到环境变化的规则；
- 由策略决定用户行为，由 LLM 负责语言实现；
- 保留当前 LLM 用户作为 Simulator-A；
- 新增规则状态机用户作为 Simulator-B；
- 用反事实和改写生成 Simulator-C。

验收标准：相同业务状态用不同表达方式测试，Agent 的逻辑分和目标完成度保持稳定。

### Phase 2：建立 SOP 执行记忆

目标：让 Agent 学习抽象业务经验，而不是记忆用户原句。

建议记录：

```json
{
  "scenario": "online_education",
  "sop_node": "step5_emotion_check",
  "failure_type": "misclassification",
  "abstract_condition": [
    "repeated_complaint",
    "negative_evaluation",
    "unresolved_for_long_time"
  ],
  "wrong_decision": "Calm",
  "correct_decision": "Dissatisfied",
  "recommended_behavior": "先识别不满，再执行 COMFORT"
}
```

验收标准：记忆对改写后的用户表达、其他 User 模型和规则用户仍然有效。

### Phase 3：实现受控自进化循环

```text
运行候选 Agent
    ↓
结构化错误归因
    ↓
生成候选策略或技能
    ↓
训练环境回归测试
    ↓
跨模拟器验证
    ↓
隐藏集验证
    ↓
通过门槛后登记为新版本
```

晋级条件建议包括：

- 总体逻辑分提升；
- 分类准确率不下降；
- 最终动作准确率不下降；
- 目标完成度提升；
- 跨模拟器得分不下降；
- 高风险和强对抗切片无严重退化。

### Phase 4：SOP 提案轨道

如果未来要让系统提出新的 SOP 节点或分支，应单独建立 SOP 演化轨道：

```text
Agent 提出 SOP 修改
    ↓
图结构和规则一致性检查
    ↓
专家或人工审核
    ↓
生成新 SOP 版本
    ↓
重新生成测试集
```

在普通 Agent 评测中不能允许 Agent 自己修改官方 SOP 或评分标准。

## 6. 原有基础修复的完成情况

下列基础工作已经完成或部分完成：

- `run_evaluation_with_llm.py` 已支持逐样本增量保存；
- 已区分 `predicted_path`、`executed_path`，并记录详细终止状态；
- 已加入 `goal_fulfillment`，并修复纯承诺话术不能直接算目标完成；
- 已完成电商退款场景的最小隐藏状态后端，而不是继续把 `system_info` 直接暴露给 Agent；
- `online_education` 和其他四个场景的真正后端环境仍未实现。

## 7. 接下来要做的事情

### P0：完成电商退款闭环的真实 API 验收

目标：确认 MVP 不只在 Fake Client 测试中成立，也能在当前 API 和目标模型上稳定运行。

具体任务：

1. 使用 `dashscope/qwen3.7-plus` 运行至少 6 个电商路径切片：未发货退款、运输中拦截、已签收换货、高信用商责、中低信用商责、低信用用户退款；
2. 记录每个样本的工具调用次数、订单号参数、工具返回、最终 Action、动作结果和最终后台状态；
3. 分别测试模型原生 `tool_calls` 与文本 JSON 工具调用两种返回格式；
4. 检查 API 不支持 tools 时的降级行为：Agent 必须询问信息或明确无法核验，不能猜测后台状态；
5. 将一次完整运行的 JSONL 结果作为回归样本，不把 API Key、完整请求头或敏感凭据写入仓库。

验收标准：每个切片都能复现固定案例；错误订单号不能触发成功动作；成功动作必须改变对应后台字段；评估器能从事件日志解释得分来源。

### P1：完善电商工具和动作契约

目标：让工具结果和业务动作接近真实客服后台，而不是只满足当前 PathList。

具体任务：

- 为订单查询增加订单不存在、订单号格式错误、重复查询、已退款等错误分支；
- 为支付查询增加支付失败、部分支付、退款处理中等状态；
- 为客户资料查询增加客户不存在、信用等级不可见、权限不足等分支；
- 为每个动作定义前置条件、幂等规则、状态转移和面向客服的错误信息；
- 明确 `submit_refund`、`intercept_shipment`、`request_document`、`exchange_order` 等动作 API，逐步将当前 finals 动作映射为正式动作调用；
- 将动作参数校验纳入 `action_execution_success` 和 `tool_argument_accuracy`，避免只比较 Action 字符串。

验收标准：所有成功/失败动作都有结构化 `ActionResult`；同一动作重复提交不会产生不合理状态；错误动作不能因为模型说了正确 Action 就得分。

### P2：迁移其余五个场景

目标：让六个场景都使用统一的后端环境接口，而不是只有电商退款具备状态闭环。

按以下顺序迁移：

1. `telecom_package`：套餐、合约、违约金、账户资格；
2. `logistics_delivery`：订单物流、保险、签收、异常件；
3. `airline_refund`：航班、舱位、会员等级、保险和退改签规则；
4. `property_service`：房屋状态、缴费状态、维修工单和紧急等级；
5. `online_education`：课程购买记录、课程状态、历史投诉、风险用户和知识回答状态。

每个场景都必须提供：

- `CaseSpec` 工厂；
- 至少两个查询工具；
- 至少一个会改变状态的业务动作；
- 用户已知事实和可披露事实；
- 后端目标状态；
- 场景专用工具/动作回归测试。

验收标准：六个场景都不需要把隐藏系统变量放进 Agent prompt；每个场景至少有一条“用户口述与后台状态不同”的测试，Agent 必须以后台核验结果为准。

### P3：重构 Customer Simulator

目标：让用户模拟器的下一句话由业务状态决定，而不是主要由关键词和满意度启发式决定。

具体任务：

- 定义统一 `UserEnvironmentState`：目标、事实、已披露事实、情绪、耐心、解决状态、升级状态；
- 把“客服询问订单号”“工具查询成功”“动作失败”“动作成功”建模为用户可观察事件；
- 用户只在客服询问或业务需要时透露订单号、客户号和凭证；
- 后端动作成功后，用户确认结果；动作失败后，用户追问、补充信息或升级投诉；
- 保留当前 LLM 用户作为自然语言实现层，同时增加规则策略用户作为稳定基线；
- 为同一 `CaseSpec` 生成多种表达、情绪和对抗强度，验证 Agent 是否依赖固定措辞。

验收标准：相同 CaseSpec 下，用户换一种说法不会改变 gold outcome；不同用户模拟器的 Agent 逻辑分差异可解释；用户不会凭空知道未查询的后台字段。

### P4：建立标准化评估与切片报告

目标：让一次评测能回答“错在哪里”，而不仅是给出一个总分。

需要新增的切片：

- 是否成功调用必要工具；
- 工具参数是否正确；
- 工具结果是否被后续决策正确使用；
- 动作是否满足前置条件；
- 动作是否造成预期后端状态转移；
- 目标是否完成；
- 是否过早结束、重复循环或超时；
- 不同情绪、对抗强度、业务状态和表达改写下的稳定性。

验收标准：每个失败样本都能关联到具体事件、错误类型和后端状态；报告同时展示 `gold_path`、`predicted_path`、`executed_path`；旧版无后端样本与新版闭环样本不能混用同一解释。

### P5：实现受控自进化，而不是直接 Prompt 自优化

目标：进化抽象业务策略和节点技能，不拟合某一个用户模拟器的措辞。

循环设计：

```text
固定 CaseSpec / 多种 User Simulator
        ↓
运行候选 Agent
        ↓
结构化错误归因
        ↓
生成候选节点技能或策略补丁
        ↓
单元测试 + 场景回归
        ↓
跨 User Simulator 验证
        ↓
隐藏集门禁
        ↓
登记 Agent 版本
```

第一批可进化对象：

- 订单号和客户身份信息的追问策略；
- 工具选择和工具参数填充策略；
- 工具结果到 SOP 节点的映射；
- 动作失败后的补救策略；
- 用户情绪变化下的沟通策略。

暂不允许自动进化：官方 SOP 图、gold labels、评分公式、隐藏测试集和后台真值生成器。

晋级门槛：总体分提升；分类、路径、动作和后台核验任一关键指标不得显著下降；跨三类用户模拟器不退化；高风险动作零越权；所有候选补丁可回滚。

### P6：最后再处理在线教育的“真正回答问题”能力

在线教育不能只把 `PLAN` 当作完成。需要额外定义：

- `answer_required`：当前问题是否要求知识回答；
- `answer_completed`：当前回复是否包含解释、步骤或示例；
- `handoff_required`：是否确实需要人工或课程资源；
- `answer_evidence`：回答中可验证的关键内容。

只有在 `answer_completed=true` 或明确完成转人工且用户目标允许转交时，才能计入目标完成。这样可以避免“已安排导师，请稍候”被误判为已经回答。
