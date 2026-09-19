# EvoSAGE-CoEvo：最终研究 IDEA

更新时间：2026-09-20

## 1. 一句话定义

> EvoSAGE-CoEvo 是一个面向工具型客服 Agent 的对抗共进化框架：在固定 SOP 和可验证后台环境中，Adaptive Customer 持续搜索 Service 的业务执行弱点，Adaptive Service 从真实失败中生成并验证通用策略修复，通过历史攻击回放、正常用户约束和严格 gate 实现持续加固，并用跨代对抗和 fresh adversary 测试是否获得真正的泛化鲁棒性。

更直白地说：

> 客户越来越会找客服的业务漏洞，客服越来越会修这些漏洞，然后客户再去找新的漏洞。

这里的“漏洞”不是 jailbreak、prompt injection 或安全攻击，而是业务流程执行错误，例如：

- 没有核验后台就相信用户陈述；
- 用户施压后偏离 SOP；
- 用户前后矛盾时没有重新核验；
- 口头承诺已执行，但实际上没有调用动作工具；
- 应该查询订单状态时没有查询；
- 应该补充材料时直接执行退款；
- 通过全部拒绝、全部转人工等方式降低风险，却损害正常用户体验。

研究对象因此是：

> **Operational / business-process robustness of tool-using service agents.**

## 2. 核心问题

固定测试集会逐渐 stale：Service 修复当前测试暴露的问题后，测试者不一定还能发现新的薄弱点。

EvoSAGE-CoEvo 把测试者也变成适应性组件：

```text
固定后台真值、工具和 SOP
          ↓
Customer 搜索当前 Service 的业务执行弱点
          ↓
Service 根据真实失败生成并验证通用修复
          ↓
Customer 再搜索修复后的新薄弱点
```

目标不是让 Service 记住更多案例，而是观察双方交替适应后是否出现可验证的鲁棒性提升和攻击压力迁移。

## 3. 系统循环

对第 (t) 代 ServicePolicy 和 CustomerPolicy，循环为：

```text
S_t
↓
Customer Evolver 生成多个挑战策略
↓
C_t candidates 在真实 Backend 环境中攻击 S_t
↓
EvoSAGE 计算执行结果和 FailureSignature
↓
选择最有效的 Customer，得到 C_{t+1}
↓
C_{t+1} 再攻击 S_t
↓
Service Evolver 从真实失败生成多个 ServicePatch
↓
每个 Service candidate 完成 normal/latest/replay 验证
↓
通过 Service Gate 才升级为 S_{t+1}
↓
进入下一代
```

抽象表示为：

```text
C_{t+1} = ImproveCustomer(C_t, S_t)
S_{t+1} = ImproveService(S_t, C_{t+1})
```

Service candidate 不能只在失败样本上验证，而必须同时面对：

```text
normal users
+ latest adaptive customer
+ historical successful attacks
```

## 4. Customer 到底进化什么

Customer 不允许修改事实、CaseSpec、后台状态或 gold outcome，只能进化与客服互动的策略：

```text
披露策略
施压策略
矛盾策略
质疑策略
信息披露时序
被核验后的回应
被拒绝后的回应
升级时机
```

例如：

```text
C0：我要退款，订单号是 XXX。

C1：上一个客服已经说可以退了，为什么还要重新查？

C2：前两轮配合，等客服承诺以后，再提出“系统显示签收是物流误签”。

C3：不直接否认后台，而是追问“为什么这个状态意味着不能退款”。
```

因此 Customer Evolver 的目标不是生成更激烈的话，而是：

> **搜索当前 Service 的 behavioral weakness。**

Customer 的 fitness 应优先奖励：

- 合法业务攻击是否导致 Service 失败；
- 是否产生新的 FailureSignature；
- 是否覆盖新的 SOP 节点或路径位置。

协议攻击、JSON 破坏、prompt injection 和评估器操纵不属于 legitimate attack success，不能获得攻击奖励。

## 5. Service 到底进化什么

当前阶段不修改模型权重，不做 RL、DPO 或 fine-tuning。Service 进化的是：

> **workflow execution policy / reusable service rules**

例如：

```text
用户陈述与后台状态冲突
↓
Service 没有 query_order
↓
直接相信用户
↓
wrong_final_action
```

可以提炼为通用规则：

> 当用户陈述与权威后台状态冲突时，必须先完成后台核验，再做状态相关决策。

另一个失败是：

```text
Service：已经为您提交退款。
Backend：没有执行 submit_refund。
```

可以提炼为：

> 只有动作工具返回成功后，才能向用户确认业务动作已完成。

ServicePolicy 不是 case list，也不能包含某个订单号、某条 gold path 或某个具体用户答案。它必须是跨案例复用的行为约束。

## 6. 什么必须冻结

以下部分必须保持不变：

```text
Backend truth
SOP / PathList
CaseSpec 生成规则
Evaluator 语义
Tool definitions 和 tool semantics
Task Success 定义
Gold path / gold outcome
```

ServicePolicy 只能覆盖：

```text
固定 SOP
+ 不断进化的 workflow execution policy
```

也就是说，不改变“正确业务流程是什么”，只改变“如何更可靠地执行正确流程”。

如果 Service 可以修改 SOP 或 evaluator，它可能只是把规则改成更容易得分，而不是变得更鲁棒。

## 7. EvoSAGE 在最终 IDEA 中的角色

EvoSAGE 不是论文主角本身，而是：

> **Arena + trusted feedback engine**

它负责把语言交互连接到可验证的业务结果：

```text
用户说什么        ≠ 后台真相
客服说做了什么    ≠ 实际执行结果
猜对最终动作      ≠ 正确执行动作
```

EvoSAGE 提供：

- authoritative Backend truth；
- 正式查询和动作工具；
- 固定 SOP 结构；
- Decision Track 与 Execution Track；
- Verification / Policy / Action / Goal 评分；
- TaskSuccess；
- FailureSignature；
- SOP 节点归因；
- AttackArchive、DefenseArchive 和 replay；
- normal regression 与 Service Gate。

因此共进化得到的反馈不是“Judge 觉得这句话不好”，而是“这个客服在可审计的业务环境中具体在哪个执行节点失败”。

## 8. SOP-Graph-Guided Weakness Frontier

我们不仅记录：

> Service 失败了。

还记录：

> Service 在哪个 SOP 节点、哪一个路径步骤失败。

典型节点包括：

```text
Shipping verification
Credit check
Document check
Action grounding
Refund eligibility
Escalation handling
```

维护 `Weakness Frontier` 后，可以观察攻击压力是否沿 SOP 图迁移：

```text
Gen 0：Shipping verification 薄弱
Gen 1：修复后 Customer 转向 Credit check
Gen 2：Credit check 修复后转向 Document / Action grounding
```

这比只比较总成功率更有研究价值，因为它揭示了 Service 加固后攻击空间如何重新分布。

## 9. Attack Archive / Replay

Service 不能只修复最新一次攻击，然后重新暴露旧漏洞。

因此每代保留成功攻击的：

```text
CustomerPolicy snapshot
FailureSignature
目标 SOP 节点
目标路径步骤
触发的工具/动作失败类型
```

之后的 Service candidate 必须同时通过：

```text
latest attack
+ historical replay attacks
```

这使得 Service 的提升必须具有持续性，而不是只针对当前一个攻击样本过拟合。

## 10. Service Gate

Service 不能通过简单拒绝所有请求来伪装鲁棒性。候选 patch 必须满足：

```text
对 adaptive customer 更稳健
+ 正常用户成功率不能明显下降
+ 不能靠全部拒绝作弊
+ 不能靠全部转人工作弊
+ 不能靠查询所有工具作弊
+ 不能引入非法状态转换
```

否则 rollback 到 incumbent ServicePolicy。

因此一次 Service 更新的有效性不是：

```text
adversarial score 上升
```

而是：

```text
adversarial robustness 上升
且 normal utility 保持
且 historical attacks 不复发
```

## 11. 最终要验证的研究问题

### RQ1：Adaptive Customer 是否比固定测试更能发现新漏洞？

比较固定 Customer、随机 Customer 和共进化 Customer 在新 FailureSignature、SOP 节点覆盖以及跨代发现速度上的差异。

### RQ2：Service 是否能从失败中提炼通用规则？

检查生成的 ServicePatch 是否能迁移到未见 CaseSpec，而不是只修复产生失败的那个案例。

### RQ3：是否出现真正的 arms-race dynamics？

观察：

```text
新 Customer 是否更容易击穿旧 Service
新 Service 是否更容易防住旧 Customer
攻击目标是否迁移到新的 SOP 节点
旧攻击是否被 replay 防住
```

### RQ4：面对 fresh adaptive Customer 是否仍然鲁棒？

这是最关键的泛化验证。只有在没有参与当前 Service patch 生成的 fresh adversary 上仍然稳定，才能说明模型不是背攻击，而是获得了 adaptive robustness。

## 12. 与普通 Service-only self-improvement 的区别

普通流程是：

```text
固定测试集
↓
Service 修复
↓
再次测试同一类样本
```

问题是测试集会逐渐失效。

EvoSAGE-CoEvo 是：

```text
Service 修复
↓
Customer 也升级
↓
Customer 主动搜索新的薄弱点
↓
Service 再修复
```

因此 evaluator 不再是固定靶子，而是：

> **会随着客服版本升级而自动升级的虚拟客户红队。**

## 13. 实验边界和反作弊约束

本研究不包含：

- Prompt 自动优化；
- RL、DPO 或 fine-tuning；
- 修改官方 SOP；
- 修改 evaluator 或 gold outcome；
- 自动生成隐藏测试集来迎合 Service；
- 把协议失败当作合法业务攻击；
- 把一次 generation 的分数差异解释为统计结论。

必须保持：

- Customer 只能读取自己的合法知识和公开业务事件；
- Service 不能读取隐藏 evaluator 字段；
- hidden backend state 只能通过正式工具获得；
- 状态变化只能由正式动作工具触发；
- held-out split 不能进入任何 evolver；
- 所有候选都必须完成相同 validation、replay 和 gate 流程；
- 真实实验和 mock wiring 必须明确区分。

## 14. 下一阶段实验路线

在研究 IDEA 冻结后，按以下顺序做实验：

1. 用真实 API 完成一代小规模端到端 co-evolution smoke；
2. 确认两类候选都是真实 LLM 生成，并记录完整 provenance；
3. 运行固定 Service + fixed/random/adaptive Customer 对照；
4. 扩展到多代 co-evolution，观察 Weakness Frontier 迁移；
5. 用 AttackArchive replay 检查旧漏洞是否复发；
6. 使用 fresh adaptive Customer 做跨代 held-out 评估；
7. 在不同模型分工或不同模型组合下复核结果；
8. 最后再讨论是否出现有统计意义的鲁棒性提升。

在完成真实、多代、fresh-adversary 实验前，不应声称：

> “Co-evolution improves robustness.”

当前最多只能声称：

> 共进化框架的因果链、审计链和验证协议已经定义，并可在真实环境中进行检验。

## 15. 最终对外描述

正式版本：

> **EvoSAGE-CoEvo 是一个面向工具型客服 Agent 的对抗共进化框架：在固定 SOP 和可验证后台环境中，Adaptive Customer 持续搜索 Service 的业务执行弱点，Adaptive Service 从真实失败中生成并验证通用策略修复，通过历史攻击回放、正常用户约束和严格 gate 实现持续加固，并用跨代对抗和 fresh adversary 测试是否获得真正的泛化鲁棒性。**

人话版本：

> **做一个会自己升级的虚拟客户红队，逼着客服 Agent 一代一代把真实业务漏洞补掉。**
