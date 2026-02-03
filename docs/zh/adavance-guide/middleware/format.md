# 格式化系统提示词

系统提示词中的占位符变量可以通过中间件进行动态解析。本库提供了两种实现方式：

1.  **全局实例 `format_prompt`**：预置了 **f-string** 风格（如 `{name}`）的格式化逻辑，适用于绝大多数场景，推荐直接使用。
2.  **中间件类 `FormatPromptMiddleware`**：当需要使用 **jinja2** 风格（如 `{{ name }}`）时，需手动实例化此类。

!!! warning "使用须知"
    若需使用 **jinja2** 风格模板，请确保已安装 `langchain-dev-utils[standard]`，详见**安装指南**。

## 变量解析顺序

占位符中的变量值遵循以下优先级从高到低的解析顺序：

1.  **优先从 `state` 中查找**：会先从 `state` 字典中查找与占位符同名的字段。
2.  **其次从 `context` 中查找**：如果在 `state` 中未找到该字段，则会继续在 `context` 对象中查找。

这意味着 `state` 中的值拥有更高的优先级，可以覆盖 `context` 中同名的值。


## 使用 f-string 风格 (`format_prompt`)

`format_prompt` 是最常用的全局实例，采用 Python 原生的 f-string 语法。以下所有示例均基于此实例。

### 仅从 `state` 中获取变量

这是最基础的用法，所有占位符变量均由 `state` 提供。

```python
from langchain_dev_utils.agents.middleware import format_prompt
from langchain.agents import AgentState

class AssistantState(AgentState):
    name: str

agent = create_agent(
    model="vllm:qwen3-4b",
    system_prompt="你是一个智能助手，你的名字叫做{name}。",
    middleware=[format_prompt],
    state_schema=AssistantState,
)

# 调用时，必须为 state 提供 'name' 的值
response = agent.invoke(
    {"messages": [HumanMessage(content="你好啊")], "name": "assistant"}
)
print(response)
```

### 同时从 `state` 和 `context` 中获取变量

以下示例展示了如何混合使用 `state` 和 `context` 的数据：

```python
from dataclasses import dataclass

@dataclass
class Context:
    user: str

agent = create_agent(
    model="vllm:qwen3-4b",
    # {name} 将从 state 获取，{user} 将从 context 获取
    system_prompt="你是一个智能助手，你的名字叫做{name}。你的使用者叫做{user}。",
    middleware=[format_prompt],
    state_schema=AssistantState,
    context_schema=Context,
)

# 调用时，为 state 提供 'name'，为 context 提供 'user'
response = agent.invoke(
    {
        "messages": [HumanMessage(content="我要去New York玩几天，帮我规划行程")],
        "name": "assistant",
    },
    context=Context(user="张三"),
)
print(response)
```

### 变量覆盖示例

当 `state` 和 `context` 中存在同名变量时，`state` 的值会优先生效。

```python
from dataclasses import dataclass

@dataclass
class Context:
    # context 中定义了 'name'
    name: str
    user: str

agent = create_agent(
    model="vllm:qwen3-4b",
    system_prompt="你是一个智能助手，你的名字叫做{name}。你的使用者叫做{user}。",
    middleware=[format_prompt],
    state_schema=AssistantState, # state 中也定义了 'name'
    context_schema=Context,
)

# 调用时，state 和 context 都提供了 'name' 的值
response = agent.invoke(
    {
        "messages": [HumanMessage(content="你叫什么名字？")],
        "name": "assistant-1",
    },
    context=Context(name="assistant-2", user="张三"),
)

# 最终的系统提示词会是 "你是一个智能助手，你的名字叫做assistant-1。你的使用者叫做张三。"
# 因为 state 的优先级更高
print(response)
```

## 使用 Jinja2 风格 (`FormatPromptMiddleware`)

如果你的系统提示词需要更复杂的逻辑（如循环、条件判断），或者习惯使用 Jinja2 语法，请使用 `FormatPromptMiddleware` 并指定 `template_format="jinja2"`。

### 基础示例

下面的示例展示了如何使用 Jinja2 语法根据条件动态生成提示词。

```python
from langchain_dev_utils.agents.middleware import FormatPromptMiddleware
from dataclasses import dataclass
from typing import Optional

@dataclass
class Context:
    user_role: Optional[str] = None  # 用户角色，例如 "VIP", "Admin"

# 手动实例化中间件，指定格式为 jinja2
jinja2_formatter = FormatPromptMiddleware(template_format="jinja2")

agent = create_agent(
    model="vllm:qwen3-4b",
    # 使用 {{ }} 语法
    system_prompt=(
        "你是一个智能助手。\n"
        "{% if user_role == 'VIP' %}"
        "请务必提供尊贵、周到的服务。\n"
        "{% elif user_role == 'Admin' %}"
        "请展示系统管理员的权限和严谨性。\n"
        "{% else %}"
        "请提供标准的用户服务。\n"
        "{% endif %}"
    ),
    middleware=[jinja2_formatter],
    context_schema=Context,
)

# 示例 1：普通用户
response = agent.invoke(
    {"messages": [HumanMessage(content="你好")]},
    context=Context(user_role="Guest"),
)

# 示例 2：VIP 用户
# 系统提示词将包含 "请务必提供尊贵、周到的服务"
response = agent.invoke(
    {"messages": [HumanMessage(content="你好")]},
    context=Context(user_role="VIP"),
)
```