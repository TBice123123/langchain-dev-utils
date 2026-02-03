
# 模型路由

`ModelRouterMiddleware` 是一个用于**根据输入内容动态路由到最适配模型**的中间件。它通过一个"路由模型"分析用户请求，从预定义的模型列表中选择最适合当前任务的模型进行处理。

### 参数说明

| 参数 | 说明 |
|------|------|
| `router_model` | 用于执行路由决策的模型。<br><br>**类型**: `str` \| `BaseChatModel`<br>**必填**: 是 |
| `model_list` | 模型配置列表。<br><br>**类型**: `list[ModelDict]`<br>**必填**: 是 |
| `router_prompt` | 自定义路由模型的提示词。<br><br>**类型**: `str`<br>**必填**: 否 |

#### `model_list` 配置说明

每个模型配置为一个字典，包含以下字段：

| 字段 | 说明 |
|------|------|
| `model_name` | 模型的唯一标识，使用 `provider:model-name` 格式。<br><br>**类型**: `str`<br>**必填**: 是 |
| `model_description` | 模型能力或适用场景的简要描述。<br><br>**类型**: `str`<br>**必填**: 是 |
| `tools` | 该模型可调用的工具白名单。<br><br>**类型**: `list[BaseTool]`<br>**必填**: 否 |
| `model_kwargs` | 模型加载时的额外参数。<br><br>**类型**: `dict`<br>**必填**: 否 |
| `model_system_prompt` | 模型的系统级提示词。<br><br>**类型**: `str`<br>**必填**: 否 |
| `model_instance` | 已实例化的模型对象。<br><br>**类型**: `BaseChatModel`<br>**必填**: 否 |


!!! tip "model_instance 字段说明"

    - **若提供**：直接使用该实例，`model_name` 仅作标识，`model_kwargs` 被忽略；适用于不使用本库的对话模型管理功能的情况。
    - **若未提供**：根据 `model_name` 和 `model_kwargs` 使用 `load_chat_model` 加载模型。
    - **命名格式**：无论哪种情况，`model_name` 的命名都推荐采用 `provider:model-name` 格式。


## 使用示例

**步骤一：定义模型列表**

```python
from langchain_dev_utils.agents.middleware.model_router import ModelDict

model_list: list[ModelDict] = [
    {
        "model_name": "vllm:qwen3-8b",
        "model_description": "适合普通任务，如对话、文本生成等",
        "model_kwargs": {
            "temperature": 0.7,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        "model_system_prompt": "你是一个助手，擅长处理普通任务，如对话、文本生成等。",
    },
    {
        "model_name": "vllm:qwen3-vl-2b",
        "model_description": "适合视觉任务",
        "tools": [],  # 如果该模型不需要任何工具，请将此字段设置为空列表 []
    },
    {
        "model_name": "vllm:qwen3-coder-flash",
        "model_description": "适合代码生成任务",
        "tools": [run_python_code],  # 仅允许使用 run_python_code 工具
    },
    {
        "model_name": "openai:gpt-4o",
        "model_description": "适合综合类高难度任务",
        "model_system_prompt": "你是一个助手，擅长处理综合类的高难度任务",
        "model_instance": ChatOpenAI(
            model="gpt-4o"
        ),  # 直接传入实例，此时 model_name 仅作标识，model_kwargs 被忽略
    },
]
```

**步骤二：创建 Agent 并启用中间件**

```python
from langchain_dev_utils.agents.middleware import ModelRouterMiddleware
from langchain_core.messages import HumanMessage

agent = create_agent(
    model="vllm:qwen3-4b",  # 此模型仅作占位，实际由中间件动态替换
    tools=[run_python_code, get_current_time],
    middleware=[
        ModelRouterMiddleware(
            router_model="vllm:qwen3-4b",
            model_list=model_list,
        )
    ],
)

# 路由中间件会根据输入内容自动选择最合适的模型
response = agent.invoke({"messages": [HumanMessage(content="帮我写一个冒泡排序代码")]})
print(response)
```

通过 `ModelRouterMiddleware`，你可以轻松构建一个多模型、多能力的 Agent，根据任务类型自动选择最优模型，提升响应质量与效率。

!!! note "并行执行"
    采用中间件实现模型路由，每次仅会分配一个任务进行执行，如果你想要将任务分解为多个子任务由多个模型进行并行执行，请参考[预置StateGraph构建函数](../graph.md)。

