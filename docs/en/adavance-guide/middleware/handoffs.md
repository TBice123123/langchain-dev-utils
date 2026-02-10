# Agent Handoff

`HandoffAgentMiddleware` is a middleware used for **flexibly switching between multiple sub-agents**. It fully implements the official LangChain `handoffs` multi-agent collaboration scheme.

## Parameter Description

| Parameter | Description |
|-----------|-------------|
| `agents_config` | A dictionary of agent configurations, where the key is the agent name and the value is the agent configuration dictionary.<br><br>**Type**: `dict[str, AgentConfig]`<br>**Required**: Yes |
| `custom_handoffs_tool_descriptions` | Custom descriptions for handoff tools, where the key is the agent name and the value is the corresponding handoff tool description.<br><br>**Type**: `dict[str, str]`<br>**Required**: No |
| `handoffs_tool_overrides` | Custom implementations for handoff tools, where the key is the agent name and the value is the corresponding handoff tool implementation.<br><br>**Type**: `dict[str, BaseTool]`<br>**Required**: No |

### `agents_config` Configuration Description

Each agent is configured as a dictionary containing the following fields:

| Field | Description |
|-------|-------------|
| `model` | Specifies the model used by this agent; if not passed, it inherits the model corresponding to the `model` parameter of `create_agent`. Supports strings (must be in `provider:model-name` format, e.g., `vllm:qwen3-4b`) or `BaseChatModel` instances.<br><br>**Type**: `str` \| `BaseChatModel`<br>**Required**: No |
| `prompt` | The system prompt for the agent.<br><br>**Type**: `str` \| `SystemMessage`<br>**Required**: Yes |
| `tools` | A list of tools the agent can call; if not passed, the agent only possesses relevant handoff tools.<br><br>**Type**: `list[BaseTool]`<br>**Required**: No |
| `default` | Whether to set as the default agent; defaults to `False`. There must be one and only one agent set to `True` in the entire configuration.<br><br>**Type**: `bool`<br>**Required**: No |
| `handoffs` | A list of names of other agents to which this agent can hand off. If set to `"all"`, it means this agent can hand off to all other agents.<br><br>**Type**: `list[str]` \| `str`<br>**Required**: Yes |


!!! note "Note"
    After using this middleware, the `tools` and `system_prompt` parameters of `create_agent` will be ignored, so there is no need to fill them in.

For this paradigm of multi-agent implementation, a tool for handoffs is often required. This middleware utilizes the `handoffs` configuration of each agent to automatically create the corresponding handoff tool for each agent. If you wish to customize the description of the handoff tool, you can achieve this via the `custom_handoffs_tool_descriptions` parameter.



## Basic Usage

In this example, four agents will be used: `time_agent`, `weather_agent`, `code_agent`, and `default_agent`.

Next, we need to create the corresponding agent configuration dictionary `agent_config`.

```python
from langchain_dev_utils.agents.middleware.handoffs import AgentConfig

agent_config: dict[str, AgentConfig] = {
    "time_agent": {
        "model": "vllm:qwen3-8b",
        "prompt": "You are a time assistant",
        "tools": [get_current_time],
        "handoffs": ["default_agent"],  # This agent can only hand off to default_agent
    },
    "weather_agent": {
        "prompt": "You are a weather assistant",
        "tools": [get_current_weather, get_current_city],
        "handoffs": ["default_agent"],
    },
    "code_agent": {
        "model": load_chat_model("vllm:qwen3-coder-flash"),
        "prompt": "You are a coding assistant",
        "tools": [
            run_code,
        ],
        "handoffs": ["default_agent"],
    },
    "default_agent": {
        "prompt": "You are an assistant",
        "default": True, # Set as the default agent
        "handoffs": "all",  # This agent can hand off to all other agents
    },
}
```

Finally, simply pass this configuration to `HandoffAgentMiddleware`.

```python
from langchain_dev_utils.agents.middleware import HandoffAgentMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    middleware=[HandoffAgentMiddleware(agents_config=agent_config)],
)

response = agent.invoke({"messages": [HumanMessage(content="What is the current time?")]})
print(response)
```
## Customizing Handoff Tool Descriptions

If you want to customize the description of the handoff tools, you can pass a second parameter `custom_handoffs_tool_descriptions`.

```python hl_lines="6-11"
agent = create_agent(
    model="vllm:qwen3-4b",
    middleware=[
        HandoffAgentMiddleware(
            agents_config=agent_config,
            custom_handoffs_tool_descriptions={
                "time_agent": "Use this tool to hand off to the time assistant to resolve time query issues",
                "weather_agent": "Use this tool to hand off to the weather assistant to resolve weather query issues",
                "code_agent": "Use this tool to hand off to the coding assistant to resolve code issues",
                "default_agent": "Use this tool to hand off to the default assistant",
            },
        )
    ],
)
```


## Customizing Handoff Tool Implementation

If you want to fully customize the implementation logic of the handoff tool, you can pass a third parameter `handoffs_tool_overrides`. Similar to the second parameter, it is also a dictionary where the key is the agent name and the value is the corresponding handoff tool implementation.

A custom handoff tool must return a `Command` object, whose `update` attribute needs to contain the `messages` key (returning the tool response) and the `active_agent` key (whose value is the name of the agent to hand off to, used to switch the current agent).

For example:

```python hl_lines="23-25"
@tool
def transfer_to_code_agent(runtime: ToolRuntime) -> Command:
    """This tool helps you hand off to the coding assistant"""
    # You can add custom logic here
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="transfer to code agent",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "active_agent": "code_agent",
            # You can add other keys to update here
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

`handoffs_tool_overrides` is used for highly customized implementations of handoff tools. If you only want to customize the description of the handoff tool, you should use `custom_handoffs_tool_descriptions`.