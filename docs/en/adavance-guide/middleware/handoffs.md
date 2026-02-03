# Agent Handoffs

`HandoffAgentMiddleware` is a middleware for **flexibly switching between multiple sub-agents**, fully implementing LangChain's official `handoffs` multi-agent collaboration solution.

## Parameter Description

| Parameter | Description |
|------|------|
| `agents_config` | Agent configuration dictionary, where keys are agent names and values are agent configuration dictionaries.<br><br>**Type**: `dict[str, AgentConfig]`<br>**Required**: Yes |
| `custom_handoffs_tool_descriptions` | Custom descriptions for handoff tools, where keys are agent names and values are the corresponding handoff tool descriptions.<br><br>**Type**: `dict[str, str]`<br>**Required**: No |
| `handoffs_tool_overrides` | Custom implementations for handoff tools, where keys are agent names and values are the corresponding handoff tool implementations.<br><br>**Type**: `dict[str, BaseTool]`<br>**Required**: No |

### `agents_config` Configuration Details

Each agent configuration is a dictionary containing the following fields:

| Field | Description |
|------|------|
| `model` | Specifies the model used by this agent; if not provided, the model from `create_agent`'s `model` parameter is used. Supports either a string (must be in `provider:model-name` format, e.g., `vllm:qwen3-4b`) or a `BaseChatModel` instance.<br><br>**Type**: `str` \| `BaseChatModel`<br>**Required**: No |
| `prompt` | The agent's system prompt.<br><br>**Type**: `str` \| `SystemMessage`<br>**Required**: Yes |
| `tools` | The list of tools callable by the agent.<br><br>**Type**: `list[BaseTool]`<br>**Required**: No |
| `default` | Whether to set as the default agent; default is `False`. Exactly one agent in the entire configuration must be set to `True`.<br><br>**Type**: `bool`<br>**Required**: No |
| `handoffs` | List of other agent names that this agent can hand off to. If set to `"all"`, it means this agent can hand off to all other agents.<br><br>**Type**: `list[str]` \| `str`<br>**Required**: Yes |

For this paradigm of multi-agent implementation, a handoff tool is often required. This middleware automatically creates corresponding handoff tools for each agent based on their `handoffs` configuration. If you want to customize the descriptions of handoff tools, you can achieve this through the `custom_handoffs_tool_descriptions` parameter.

## Basic Usage

In this example, we will use four agents: `time_agent`, `weather_agent`, `code_agent`, and `default_agent`.

Next, we'll create the corresponding agent configuration dictionary `agent_config`.

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
        "prompt": "You are a code assistant",
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

Finally, pass this configuration to `HandoffAgentMiddleware`.

```python
from langchain_dev_utils.agents.middleware import HandoffAgentMiddleware

agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[
        get_current_time,
        get_current_weather,
        get_current_city,
        run_code,
    ],
    middleware=[HandoffAgentMiddleware(agents_config=agent_config)],
)

response = agent.invoke({"messages": [HumanMessage(content="What is the current time?")]})
print(response)
```

## Customizing Handoff Tool Descriptions

If you want to customize the descriptions of handoff tools, you can pass the second parameter `custom_handoffs_tool_descriptions`.

```python
agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[
        get_current_time,
        get_current_weather,
        get_current_city,
        run_code,
    ],
    middleware=[
        HandoffAgentMiddleware(
            agents_config=agent_config,
            custom_handoffs_tool_descriptions={
                "time_agent": "This tool is used to hand off to the time assistant to solve time query problems",
                "weather_agent": "This tool is used to hand off to the weather assistant to solve weather query problems",
                "code_agent": "This tool is used to hand off to the code assistant to solve code problems",
                "default_agent": "This tool is used to hand off to the default assistant",
            },
        )
    ],
)
```

## Customizing Handoff Tool Implementation

If you want to completely customize the logic of handoff tools, you can pass the third parameter `handoffs_tool_overrides`. Similar to the second parameter, it is also a dictionary where keys are agent names and values are the corresponding handoff tool implementations.

A custom handoff tool must return a `Command` object, whose `update` attribute must include a `messages` key (returning the tool response) and an `active_agent` key (with the value being the name of the agent to hand off to, used to switch the current agent).

For example:

```python
@tool
def transfer_to_code_agent(runtime: ToolRuntime) -> Command:
    """This tool helps you transfer to the code agent."""
    # You can add custom logic here
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="Transfer to code agent",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "active_agent": "code_agent",
            # You can add other keys to update here
        }
    )

agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[
        get_current_time,
        get_current_weather,
        get_current_city,
        run_code,
    ],
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

`handoffs_tool_overrides` is used for highly customized implementations of handoff tools. If you only want to customize the description of handoff tools, you should use `custom_handoffs_tool_descriptions`.

