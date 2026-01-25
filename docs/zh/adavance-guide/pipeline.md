# 状态图编排

## 概述

在 LangGraph 中，状态图编排是构建复杂 AI 应用的关键能力。通过按特定模式组合多个状态图，可以形成强大且逻辑清晰的工作流。

本库提供两种最常见的编排方式：

| 编排方式 | 功能描述 | 适用场景 |
|----------|----------|----------|
| **顺序编排** | 按顺序组合多个状态图，形成顺序工作流 | 任务需按步骤执行且依赖前一步输出 |
| **并行编排** | 并行组合多个状态图，形成并行工作流 | 多个任务相互独立，可同时执行以提高效率 |

## 顺序编排

顺序编排（Sequential Pipeline）将复杂任务拆解为连续、有序的子任务，并交由不同的专门化智能体依次处理。

使用 `create_sequential_pipeline` 可将多个状态图以顺序方式组合。

### 典型应用场景

以用户购买商品为例，典型流程如下：

```mermaid
graph LR
    Start([用户下单请求])
    Inv[库存确认]
    Ord[创建订单]
    Pay[完成支付]
    Del[确认发货]
    End([订单完成])

    Start --> Inv --> Ord --> Pay --> Del --> End
```

该流程环环相扣，顺序不可颠倒。

其中，库存确认、创建订单、完成支付、确认发货四个步骤分别由专门智能体处理。通过 `create_sequential_pipeline` 将四个状态图顺序编排，即可形成高度自动化、职责清晰的商品购买工作流。


### 基础示例

下面示例展示如何用 `create_sequential_pipeline` 构建商品购买的顺序工作流：


先创建对话模型对象。这里以接入本地 vllm 部署的 `qwen3-4b` 为例，其接口与 OpenAI 兼容，因此可直接用 `create_openai_compatible_model` 构建模型类。

```python
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    base_url="http://localhost:8000/v1",
    chat_model_cls_name="ChatVLLM",
)
```
再实例化一个 `ChatVLLM` 对象，供后续智能体调用。

```python
model = ChatVLLM(model="qwen3-4b")
```
随后创建相关工具，例如查询库存、创建订单、进行支付等。

??? example "工具的实现参考"

    ```python
    from langchain_core.tools import tool

    @tool
    def check_inventory(product_name: str) -> dict:
        """查询库存"""
        return {"product_name": product_name, "in_stock": True, "available": 42}

    @tool
    def create_order(product_name: str, quantity: int) -> str:
        """创建订单"""
        return f"已创建订单 ORD-10001，商品：{product_name}，数量：{quantity}。"

    @tool
    def pay_order(order_id: str) -> str:
        """支付订单"""
        return f"订单 {order_id} 支付成功。"

    @tool
    def confirm_delivery(order_id: str, address: str) -> str:
        """确认发货"""
        return f"订单 {order_id} 已安排发货，收货地址：{address}。"
    ```

然后创建对应的四个子智能体。

```python
from langchain.agents import create_agent

inventory_agent = create_agent(
    model=model,
    tools=[check_inventory],
    system_prompt="你是库存助手，负责确认商品是否有货。",
    name="inventory_agent",
)

order_agent = create_agent(
    model=model,
    tools=[create_order],
    system_prompt="你是下单助手，负责创建订单。",
    name="order_agent",
)

payment_agent = create_agent(
    model=model,
    tools=[pay_order],
    system_prompt="你是支付助手，负责完成支付。",
    name="payment_agent",
)

delivery_agent = create_agent(
    model=model,
    tools=[confirm_delivery],
    system_prompt="你是发货助手，负责确认发货信息。",
    name="delivery_agent",
)
```

最后使用 `create_sequential_pipeline` 将这四个智能体按顺序编排。

```python
from langchain_dev_utils.pipeline import create_sequential_pipeline
from langchain.agents import AgentState

graph = create_sequential_pipeline(
    sub_graphs=[
        inventory_agent,
        order_agent,
        payment_agent,
        delivery_agent,
    ],
    state_schema=AgentState,
)
```
运行测试：

```python
response = graph.invoke(
    {
        "messages": [
            HumanMessage("我要买一副无线耳机，数量2，请下单，收货地址X市X区X路X号")
        ]
    }
)
print(response)
```

### 上下文工程优化

该示例会把前面所有智能体的完整上下文依次传给当前智能体，可能造成上下文膨胀，影响性能和效果。

可采用以下方案精简上下文：

| 方案 | 描述 | 优点 |
|------|------|------|
| **使用中间件** | 使用 `create_agent` 配合中间件，仅抽取并传递必要信息 | 实现简单，代码改动小 |
| **自定义状态图** | 基于 `LangGraph` 自定义状态图，显式控制状态字段与消息流动 | 灵活性高，可精确控制 |

??? example "点击查看利用中间件解决的参考代码"

    ```python
    from typing import Any

    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import RemoveMessage
    from langgraph.runtime import Runtime

    from langchain_dev_utils.agents.middleware import format_prompt


    class PurchaseState(AgentState, total=False):
        stock: str
        order: str
        payment: str
        delivery: str


    class ClearAgentContextMiddleware(AgentMiddleware):
        state_schema = PurchaseState

        def __init__(self, result_save_key: str) -> None:
            super().__init__()
            self.result_save_key = result_save_key

        def after_agent(
            self, state: PurchaseState, runtime: Runtime
        ) -> dict[str, Any] | None:
            final_message = state["messages"][-1]
            update_key = self.result_save_key
            return {
                "messages": [
                    RemoveMessage(id=msg.id or "") for msg in state["messages"][1:]
                ],
                update_key: final_message.content,
            }


    inventory_agent = create_agent(
        model=model,
        tools=[check_inventory],
        system_prompt="你是库存助手，负责确认商品是否有货。最终请输出库存查询结果。",
        name="inventory_agent",
        state_schema=PurchaseState,
        middleware=[format_prompt, ClearAgentContextMiddleware("stock")],
    )

    order_agent = create_agent(
        model=model,
        tools=[create_order],
        system_prompt=(
            "你是下单助手，负责创建订单。\n"
            "库存结果：{stock}\n"
            "请基于库存结果创建订单，并输出订单号。"
        ),
        name="order_agent",
        state_schema=PurchaseState,
        middleware=[format_prompt, ClearAgentContextMiddleware("order")],
    )

    payment_agent = create_agent(
        model=model,
        tools=[pay_order],
        system_prompt=(
            "你是支付助手，负责完成支付。\n"
            "订单结果：{order}\n"
            "请从订单结果中获取订单号并完成支付。"
        ),
        name="payment_agent",
        state_schema=PurchaseState,
        middleware=[format_prompt, ClearAgentContextMiddleware("payment")],
    )

    delivery_agent = create_agent(
        model=model,
        tools=[confirm_delivery],
        system_prompt=(
            "你是发货助手，负责确认发货信息。\n"
            "订单结果：{order}\n"
            "支付结果：{payment}\n"
            "请确认发货并复述收货地址。"
        ),
        name="delivery_agent",
        state_schema=PurchaseState,
        middleware=[format_prompt, ClearAgentContextMiddleware("delivery")],
    )

    graph = create_sequential_pipeline(
        sub_graphs=[
            inventory_agent,
            order_agent,
            payment_agent,
            delivery_agent,
        ],
        state_schema=PurchaseState,
    )

    response = graph.invoke(
        {
            "messages": [
                HumanMessage("我要买一副无线耳机，数量2，请下单，收货地址X市X区X路X号")
            ]
        }
    )
    print(response)
    ```

    **实现说明**：

    1. **扩展状态模式**：在智能体的 State Schema 中新增 `stock`、`order`、`payment`、`delivery` 四个字段，用于保存各智能体的最终输出。

    2. **自定义中间件**：创建 `ClearAgentContextMiddleware` 中间件，在每个智能体结束后先用 `RemoveMessage` 清理上下文，再将最终结果（`final_message.content`）写入对应字段。

    3. **动态提示格式化**：使用 `format_prompt` 中间件，在运行时按需将前置输出拼入 `system_prompt`。



!!! info "提示"

    对于串行组合的图，LangGraph 的 `StateGraph` 提供 `add_sequence` 作为简写，更适合节点为函数（而非子图）的场景。

    ```python
    graph = StateGraph(AgentState)
    graph.add_sequence([("graph1", graph1), ("graph2", graph2), ("graph3", graph3)])
    graph.add_edge("__start__", "graph1")
    graph = graph.compile()
    ```

    但上述写法仍略显繁琐，更推荐使用 `create_sequential_pipeline`，一行代码即可快速构建串行执行图。

## 并行编排

并行编排（Parallel Pipeline）将多个状态图并行组合，并发执行各任务，从而提高执行效率。

使用 `create_parallel_pipeline` 可将多个状态图以并行方式组合，实现并行执行。

### 典型应用场景

在商品购买场景中，用户可能同时需要多种查询，例如商品详情、库存、优惠与运费估算，可并行执行。

流程如下：

```mermaid
graph LR
    Start([用户请求])
    
    subgraph Parallel [并行执行]
        direction TB
        Prod[商品详情查询]
        Inv[库存查询]
        Prom[优惠计算]
        Ship[运费估算]
    end
    
    End([聚合结果])

    Start --> Prod
    Start --> Inv
    Start --> Prom
    Start --> Ship

    Prod --> End
    Inv --> End
    Prom --> End
    Ship --> End
```


### 基础示例

先创建几个工具。

??? example "工具的实现参考"

    ```python
    @tool
    def get_product_detail(product_name: str) -> dict:
        """查询商品详情"""
        return {
            "product_name": product_name,
            "sku": "SKU-10001",
            "price": 299,
            "highlights": ["主动降噪", "蓝牙5.3", "30小时续航"],
        }

    @tool
    def check_inventory(product_name: str) -> dict:
        """查询库存"""
        return {"product_name": product_name, "in_stock": True, "available": 42}

    @tool
    def calculate_promotions(product_name: str, quantity: int) -> dict:
        """计算优惠"""
        return {
            "product_name": product_name,
            "quantity": quantity,
            "discounts": ["满300减30", "会员95折"],
            "estimated_discount": 45,
        }

    @tool
    def estimate_shipping(address: str) -> dict:
        """估算运费和时效"""
        return {
            "address": address,
            "fee": 12,
            "eta_days": 2,
        }
    ```

以及对应的子智能体：

```python
product_agent = create_agent(
    model,
    tools=[get_product_detail],
    system_prompt="你是商品助理，负责解析用户需求并查询商品详情。",
    name="product_agent",
    state_schema=AgentState,
)

inventory_agent = create_agent(
    model,
    tools=[check_inventory],
    system_prompt="你是库存助理，负责根据SKU查询库存。",
    name="inventory_agent",
    state_schema=AgentState,
)

promotion_agent = create_agent(
    model,
    tools=[calculate_promotions],
    system_prompt="你是优惠助理，负责计算当前可用优惠和预计折扣。",
    name="promotion_agent",
    state_schema=AgentState,
)

shipping_agent = create_agent(
    model,
    tools=[estimate_shipping],
    system_prompt="你是配送助理，负责估算运费和时效。",
    name="shipping_agent",
    state_schema=AgentState,
)
```

用 `create_parallel_pipeline` 组合子智能体。

```python
from langchain_dev_utils.pipeline import create_parallel_pipeline

graph = create_parallel_pipeline(
    sub_graphs=[
        product_agent,
        inventory_agent,
        promotion_agent,
        shipping_agent,
    ],
    state_schema=AgentState,
)
```
运行测试：

```python
response = graph.invoke(
    {"messages": [HumanMessage("我想买一副无线耳机，数量2，收货地址X市X区X路X号")]}
)
print(response)
```


### 使用分支函数指定并行执行的子图

在多数场景下，不希望所有子图都并行执行，而是按条件并行部分子图。此时需使用 `branches_fn` 指定分支函数。分支函数需返回 `Send` 列表，每个 `Send` 包含子图名称与输入。

#### 应用场景

`Router` 是多智能体系统的典型架构：由路由模型根据用户请求进行需求分析与任务拆解，再分发给若干业务智能体执行。在订单查询场景中，用户可能同时关心订单状态、商品信息或退款，此时可由路由模型将请求分配给订单、商品、退款等智能体。

先编写工具。

??? example "工具的实现参考"

    ```python
    @tool
    def list_orders() -> dict:
        """查询用户订单列表"""
        return {
            "orders": [
                {
                    "order_id": "ORD-20250101-0001",
                    "status": "已发货",
                    "items": [{"product_name": "无线耳机", "qty": 1}],
                    "created_at": "2025-01-01 10:02:11",
                },
                {
                    "order_id": "ORD-20241215-0234",
                    "status": "已完成",
                    "items": [{"product_name": "机械键盘", "qty": 1}],
                    "created_at": "2024-12-15 21:18:03",
                },
            ],
        }

    @tool
    def get_order_detail(order_id: str) -> dict:
        """查询订单详情"""
        return {
            "status": "已发货",
            "receiver": {"name": "张三", "phone": "138****0000"},
            "items": [
                {
                    "product_id": "P-10001",
                    "product_name": "无线耳机",
                    "qty": 1,
                    "price": 299,
                }
            ],
        }

    @tool
    def get_shipping_trace(tracking_no: str) -> dict:
        """查询物流轨迹"""
        return {
            "events": [
                {"time": "2025-01-02 09:10", "status": "快件已揽收"},
                {"time": "2025-01-02 18:45", "status": "快件运输中"},
                {"time": "2025-01-03 11:20", "status": "快件已到达派送站"},
            ],
        }

    @tool
    def search_products(query: str) -> dict:
        """搜索产品"""
        return {
            "results": [
                {
                    "product_id": "P-10001",
                    "name": "无线耳机 Pro",
                    "price": 299,
                    "highlights": ["降噪", "蓝牙5.3", "续航30小时"],
                },
                {
                    "product_id": "P-10002",
                    "name": "无线耳机 Lite",
                    "price": 199,
                    "highlights": ["轻量", "低延迟", "续航24小时"],
                },
            ],
        }

    @tool
    def get_product_detail(product_id: str) -> dict:
        """查询产品详情"""
        return {
            "product_id": product_id,
            "name": "无线耳机 Pro",
            "price": 299,
            "specs": {"color": ["黑", "白"], "warranty_months": 12},
            "description": "主打降噪与长续航的真无线耳机。",
        }


    @tool
    def check_inventory(product_name: str) -> dict:
        """查询库存"""
        return {"product_name": product_name, "in_stock": True, "available": 42}

    @tool
    def create_refund(order_id: str, reason: str) -> dict:
        """发起退款"""
        return {
            "refund_id": "RFD-20250103-0009",
            "status": "已提交",
            "reason": reason,
            "estimated_days": 3,
        }

    @tool
    def get_refund_status(refund_id: str) -> dict:
        """查询退款状态"""
        return {
            "refund_id": refund_id,
            "status": "处理中",
            "progress": [
                {"time": "2025-01-03 12:05", "status": "已提交"},
                {"time": "2025-01-03 12:20", "status": "客服审核中"},
            ],
            "estimated_days": 2,
        }

    @tool
    def refund_policy() -> dict:
        """查看退款政策"""
        return {
            "window_days": 7,
            "requirements": ["商品完好", "配件齐全", "提供订单号"],
            "notes": ["部分活动商品不支持无理由退款", "到账时间视支付渠道而定"],
        }
    ```

然后创建对应的子智能体。

```python
ORDER_AGENT_PROMPT = (
    "你是订单管理助手。\n"
    "你可以使用工具来查询订单列表、订单详情、物流轨迹。\n"
    "优先使用工具获取信息，再基于工具结果给出结论。\n"
    "输出要求：用中文回答，结构清晰，必要时用条目列出订单信息。\n"
)

order_agent = create_agent(
    model,
    system_prompt=ORDER_AGENT_PROMPT,
    tools=[list_orders, get_order_detail, get_shipping_trace],
    name="order_agent",
)


PRODUCT_AGENT_PROMPT = (
    "你是产品管理助手。\n"
    "你可以使用工具来搜索产品、查看产品详情、查询库存。\n"
    "优先使用工具获取信息，再基于工具结果给出建议。\n"
    "当用户的需求不明确时，先提出一个澄清问题（例如品类/预算/用途）。\n"
    "输出要求：用中文回答，给出可执行的下一步建议。\n"
)


product_agent = create_agent(
    model,
    system_prompt=PRODUCT_AGENT_PROMPT,
    tools=[search_products, get_product_detail, check_inventory],
    name="product_agent",
)


REFUND_AGENT_PROMPT = (
    "你是退款管理助手。\n"
    "你可以使用工具来发起退款、查询退款状态、查看退款政策。\n"
    "优先使用工具获取信息；若用户缺少关键字段（例如订单号），先追问。\n"
    "输出要求：用中文回答，明确告知退款进度/所需材料/预计时间。\n"
)


refund_agent = create_agent(
    model,
    system_prompt=REFUND_AGENT_PROMPT,
    tools=[create_refund, get_refund_status, refund_policy],
    name="refund_agent",
)
```

再编写分支函数：由路由模型根据请求返回要执行的智能体名称及对应任务描述。

```python
from typing import Literal, cast
from typing_extensions import TypedDict


from langchain_core.messages import SystemMessage
from langgraph.types import Send
from pydantic import BaseModel, Field


class RouterInput(TypedDict):
    query: str


class RouterState(AgentState):
    query: str


ROUTER_SYSTEM_PROMPT = (
    "你是一个Router模型，只负责把用户问题拆分并分发到合适的业务子智能体。\n"
    "可选的业务域只有：order（订单）、product（产品）、refund（退款）。\n"
    "你必须输出一个 classifications 列表（用于并行调用多个子智能体）。\n"
    "规则：\n"
    "1) source 必须是上述三个之一；\n"
    "2) query 必须是发给该子智能体的、可直接执行的任务描述；\n"
    "3) 如果用户一句话中同时涉及多个业务域（例如‘查订单’+‘看产品’+‘问退款’），必须拆成多个 classification，以便并行执行；\n"
    "4) 如果无法判断，优先选择 product，并把问题原样交给它。\n"
    "示例A：用户：‘查一下ORD-1物流，并看看这款耳机有没有货’ -> 返回2条：order(查询物流)+product(查询库存)。\n"
    "示例B：用户：‘我想退ORD-1，退款多久到账’ -> 返回1条：refund(发起/查询退款)。\n"
    "示例C：用户：‘我想知道这款耳机的规格’ -> 返回1条：product(查询详情)。\n"
)


class Classification(TypedDict):
    """一次路由决策：调用哪个智能体并附带什么查询。"""

    source: Literal["order", "refund", "product"]
    query: str


class ClassificationResult(BaseModel):
    """将用户查询分类为面向智能体的子问题的结果。"""

    classifications: list[Classification] = Field(
        description="要调用的智能体列表及其对应的子问题"
    )


def branch_fn(state: RouterState) -> list[Send]:
    structured_llm = model.with_structured_output(ClassificationResult)

    query = state.get("query")
    classification_result = cast(
        ClassificationResult,
        structured_llm.invoke(
            [
                SystemMessage(ROUTER_SYSTEM_PROMPT),
                HumanMessage(query),
            ]
        ),
    )

    classifications = classification_result.classifications or []
    if not classifications:
        classifications = [{"source": "product", "query": query}]

    sends: list[Send] = []
    for res in classifications:
        source = res.get("source")
        if source not in {"order", "refund", "product"}:
            source = "product"
        sub_query = (res.get("query") or query).strip() or query
        sends.append(Send(f"{source}_agent", {"messages": [HumanMessage(sub_query)]}))
    return sends
```

最终使用 `create_parallel_agent` 创建并行智能体，并传入分支函数。


运行测试：

```python
response_single = graph.invoke(
    {
        "query": "你好，我要查询一下之前购买的产品",
    }
)
print(response_single)

response_parallel = graph.invoke(
    {
        "query": "推荐一款适合通勤的无线耳机并看看库存；同时，告诉我你们商品的退款政策？",
    }
)
print(response_parallel)
```


!!! tip "提示"

    - **未传入 `branches_fn` 参数时**：所有子图都会并行执行
    - **传入 `branches_fn` 参数时**：执行哪些子图由该函数的返回值决定
