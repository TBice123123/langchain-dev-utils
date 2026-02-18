# Agent Handoff

`HandoffAgentMiddleware` is a middleware designed for **flexible switching between multiple Agents**, fully implementing LangChain's official `handoffs` multi-agent collaboration scheme.

## Parameter Description

| Parameter | Description |
|------|------|
| `agents_config` | Agent configuration dictionary where keys are agent names and values are agent configuration dictionaries.<br><br>**Type**: `dict[str, AgentConfig]`<br>**Required**: Yes |
| `custom_handoffs_tool_descriptions` | Custom descriptions for handoff tools, where keys are agent names and values are the corresponding handoff tool descriptions.<br><br>**Type**: `dict[str, str]`<br>**Required**: No |
| `handoffs_tool_overrides` | Custom implementations for handoff tools, where keys are agent names and values are the corresponding handoff tool implementations.<br><br>**Type**: `dict[str, BaseTool]`<br>**Required**: No |

### `agents_config` Configuration Details

Each agent configuration is a dictionary containing the following fields:

| Field | Description |
|------|------|
| `model` | Specifies the model used by the agent; if not provided, the model corresponding to the `model` parameter in `create_agent` will be used. Supports strings (must be in `provider:model-name` format, e.g., `vllm:qwen2.5-7b`) or a `BaseChatModel` instance.<br><br>**Type**: `str` \| `BaseChatModel`<br>**Required**: No |
| `prompt` | The system prompt for the agent.<br><br>**Type**: `str` \| `SystemMessage`<br>**Required**: Yes |
| `tools` | List of tools the agent can call; if not provided, the agent will only possess relevant handoff tools.<br><br>**Type**: `list[BaseTool]`<br>**Required**: No |
| `default` | Whether to set this agent as the default; defaults to `False`. Exactly one agent in the configuration must be set to `True`.<br><br>**Type**: `bool`<br>**Required**: No |
| `handoffs` | List of other agent names this agent can hand off to. If set to `"all"`, it indicates the agent can hand off to all other agents.<br><br>**Type**: `list[str]` \| `str`<br>**Required**: Yes |


!!! note "Note"
    When using this middleware, the `tools` and `system_prompt` parameters of `create_agent` are ignored, so there is no need to fill them.

For this paradigm of multi-agent implementation, a tool for handoffs is often required. This middleware automatically creates corresponding handoff tools for each agent using the `handoffs` configuration. To customize the description of the handoff tools, you can use the `custom_handoffs_tool_descriptions` parameter.

## Basic Usage

In this example, four agents will be used: `time_agent`, `weather_agent`, `code_agent`, and `default_agent`.

Next, create the configuration dictionary `agent_config` for the corresponding agents.

```python
from langchain_dev_utils.agents.middleware.handoffs import AgentConfig

agent_config: dict[str, AgentConfig] = {
    "time_agent": {
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
        "model": load_chat_model("vllm:glm-4.7-flash"),
        "prompt": "You are a code assistant",
        "tools": [
            run_code,
        ],
        "handoffs": ["default_agent"],
    },
    "default_agent": {
        "model": "openai:gpt-4o-mini",
        "prompt": "You are an assistant",
        "default": True, # Set as default agent
        "handoffs": "all",  # This agent can hand off to all other agents
    },
}
```

Finally, pass this configuration to `HandoffAgentMiddleware`.

```python
from langchain_dev_utils.agents.middleware import HandoffAgentMiddleware

agent = create_agent(
    model="vllm:qwen2.5-7b",
    middleware=[HandoffAgentMiddleware(agents_config=agent_config)],
)

response = agent.invoke({"messages": [HumanMessage(content="What is the current time?")]})
print(response)
```

## Custom Handoff Tool Descriptions

If you want to customize the description of the handoff tools, you can pass the second parameter `custom_handoffs_tool_descriptions`.

```python hl_lines="6-11"
agent = create_agent(
    model="vllm:qwen2.5-7b",
    middleware=[
        HandoffAgentMiddleware(
            agents_config=agent_config,
            custom_handoffs_tool_descriptions={
                "time_agent": "This tool is used to hand off to the time assistant to solve time queries",
                "weather_agent": "This tool is used to hand off to the weather assistant to solve weather queries",
                "code_agent": "This tool is used to hand off to the code assistant to solve code issues",
                "default_agent": "This tool is used to hand off to the default assistant",
            },
        )
    ],
)
```


## Custom Handoff Tool Implementation

If you want to fully customize the logic of the handoff tool, you can pass the third parameter `handoffs_tool_overrides`. Similar to the second parameter, it is also a dictionary where keys are agent names and values are the corresponding handoff tool implementations.

A custom handoff tool must return a `Command` object, where the `update` attribute needs to contain a `messages` key (returning the tool response) and an `active_agent` key (the value is the name of the agent to hand off to, used to switch the current agent).

For example:

```python hl_lines="23-25"
@tool
def transfer_to_code_agent(runtime: ToolRuntime) -> Command:
    """This tool helps you hand off to the code assistant"""
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
    model="vllm:qwen2.5-7b",
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

`handoffs_tool_overrides` is used for highly customizing the implementation of the handoff tool. If you only want to customize the description of the handoff tool, you should use `custom_handoffs_tool_descriptions`.