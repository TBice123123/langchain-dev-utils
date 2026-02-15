# Overview

Middleware components are pluggable modules specifically designed for LangChain's pre-built Agents, aiming to achieve fine-grained control over the Agent's internal behavior. In addition to the built-in middleware provided by the LangChain framework, this library further enriches middleware support based on practical application scenarios.

The middleware provided by this library includes:

- [`PlanMiddleware`](plan.md): Task planning, breaking down complex tasks into ordered subtasks
- [`ModelRouterMiddleware`](router.md): Dynamically routing input to the most suitable model based on content
- [`HandoffAgentMiddleware`](handoffs.md): Flexibly handing off tasks between multiple sub-agents
- [`ToolCallRepairMiddleware`](tool-call-repair.md): Automatically repairing invalid tool calls from LLMs
- [`FormatPromptMiddleware`](format.md): Dynamically formatting placeholders in system prompts

Furthermore, this library extends the functionality of official middleware, enhancing model configuration usability by supporting model specification through string parameters:

- SummarizationMiddleware
- LLMToolSelectorMiddleware
- ModelFallbackMiddleware
- LLMToolEmulator


!!! note "Note"

    In the subsequent examples, we import the `create_agent` function from `langchain_dev_utils.agents` rather than `langchain.agents`. This is because this library provides a function that is functionally identical to the official `create_agent`, but has been extended to support specifying models via strings. This allows for the direct use of models registered via `register_model_provider` without the need to initialize and pass in a model instance.

    Before running the examples, please ensure:

    1.Register the `vllm` model provider

    ```python
    register_model_provider(
        "vllm",
        "openai-compatible",
        base_url="http://localhost:8000/v1",
    )
    ```

    2.Import the `create_agent` function from `langchain_dev_utils.agents`

    ```python
    from langchain_dev_utils.agents import create_agent
    ```
