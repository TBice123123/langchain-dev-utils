# 智能体交接

`HandoffAgentMiddleware` 是一个用于**在多个子 Agent 之间灵活切换**的中间件，完整实现了 LangChain 官方的 `handoffs` 多智能体协作方案。

## 参数说明

| 参数 | 说明 |
|------|------|
| `agents_config` | 智能体配置字典，键为智能体名称，值为智能体配置字典。<br><br>**类型**: `dict[str, AgentConfig]`<br>**必填**: 是 |
| `custom_handoffs_tool_descriptions` | 自定义交接工具的描述，键为智能体名称，值为对应的交接工具描述。<br><br>**类型**: `dict[str, str]`<br>**必填**: 否 |
| `handoffs_tool_overrides` | 自定义交接工具的实现，键为智能体名称，值为对应的交接工具实现。<br><br>**类型**: `dict[str, BaseTool]`<br>**必填**: 否 |

### `agents_config` 配置说明

每个智能体配置为一个字典，包含以下字段：

| 字段 | 说明 |
|------|------|
| `model` | 指定该智能体使用的模型；若不传，则沿用 `create_agent` 的 `model` 参数对应的模型。支持字符串（须为 `provider:model-name` 格式，如 `vllm:qwen3-4b`）或 `BaseChatModel` 实例。<br><br>**类型**: `str` \| `BaseChatModel`<br>**必填**: 否 |
| `prompt` | 智能体的系统提示词。<br><br>**类型**: `str` \| `SystemMessage`<br>**必填**: 是 |
| `tools` | 智能体可调用的工具列表；若不传，该智能体仅拥有相关的交接工具。<br><br>**类型**: `list[BaseTool]`<br>**必填**: 否 |
| `default` | 是否设为默认智能体；缺省为 `False`。全部配置中必须且只能有一个智能体设为 `True`。<br><br>**类型**: `bool`<br>**必填**: 否 |
| `handoffs` | 该智能体可交接给的其它智能体名称列表。若设为 `"all"`，则表示该智能体可交接给所有其它智能体。<br><br>**类型**: `list[str]` \| `str`<br>**必填**: 是 |


!!! note "注意"
    使用本中间件后，`create_agent` 的 `tools` 与 `system_prompt` 参数会被忽略，故无需填写。

对于这种范式的多智能体实现，往往需要一个用于交接（handoffs）的工具。本中间件利用每个智能体的 `handoffs` 配置，自动为每个智能体创建对应的交接工具。如果要自定义交接工具的描述，则可以通过 `custom_handoffs_tool_descriptions` 参数实现。



## 基础用法

本示例中，将使用四个智能体：`time_agent`、`weather_agent`、`code_agent` 和 `default_agent`。

接下来要创建对应智能体的配置字典 `agent_config`。

```python
from langchain_dev_utils.agents.middleware.handoffs import AgentConfig

agent_config: dict[str, AgentConfig] = {
    "time_agent": {
        "model": "vllm:qwen3-8b",
        "prompt": "你是一个时间助手",
        "tools": [get_current_time],
        "handoffs": ["default_agent"],  # 该智能体只能交接到default_agent
    },
    "weather_agent": {
        "prompt": "你是一个天气助手",
        "tools": [get_current_weather, get_current_city],
        "handoffs": ["default_agent"],
    },
    "code_agent": {
        "model": load_chat_model("vllm:qwen3-coder-flash"),
        "prompt": "你是一个代码助手",
        "tools": [
            run_code,
        ],
        "handoffs": ["default_agent"],
    },
    "default_agent": {
        "prompt": "你是一个助手",
        "default": True, # 设为默认智能体
        "handoffs": "all",  # 该智能体可以交接到所有其它智能体
    },
}
```

最终将这个配置传递给 `HandoffAgentMiddleware` 即可。

```python
from langchain_dev_utils.agents.middleware import HandoffAgentMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    middleware=[HandoffAgentMiddleware(agents_config=agent_config)],
)

response = agent.invoke({"messages": [HumanMessage(content="当前时间是多少？")]})
print(response)
```
## 自定义交接工具描述

如果想要自定义交接工具的描述，可以传递第二个参数 `custom_handoffs_tool_descriptions`。

```python hl_lines="12-17"
agent = create_agent(
    model="vllm:qwen3-4b",
    middleware=[
        HandoffAgentMiddleware(
            agents_config=agent_config,
            custom_handoffs_tool_descriptions={
                "time_agent": "此工具用于交接到时间助手去解决时间查询问题",
                "weather_agent": "此工具用于交接到天气助手去解决天气查询问题",
                "code_agent": "此工具用于交接到代码助手去解决代码问题",
                "default_agent": "此工具用于交接到默认的助手",
            },
        )
    ],
)
```


## 自定义交接工具实现

如果你想完全自定义实现交接工具的逻辑，则可以传递第三个参数 `handoffs_tool_overrides`。与第二个参数类似，它也是一个字典，键为智能体名称，值为对应的交接工具实现。

自定义交接工具必须返回一个 `Command` 对象，其 `update` 属性需包含 `messages` 键（返回工具响应）和 `active_agent` 键（值为要交接的智能体名称，用于切换当前智能体）。

例如：

```python hl_lines="29-31"
@tool
def transfer_to_code_agent(runtime: ToolRuntime) -> Command:
    """此工具帮助你交接到代码助手"""
    #这里你可以添加自定义逻辑
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="transfer to code agent",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "active_agent": "code_agent",
            #这里你可以添加其它的要更新的键
        }
    )

agent = create_agent(
    model="vllm:qwen3-4b",
    middleware=[
        HandoffAgentMiddleware(
            agents_config=agent_config,
            handoffs_tool_overrides={
                "code_agent": transfer_to_code_agent,
            },
        )
    ],
)
```

`handoffs_tool_overrides` 用于高度定制化交接工具的实现，如果仅仅是想要自定义交接工具的描述，则应该使用 `custom_handoffs_tool_descriptions`。

