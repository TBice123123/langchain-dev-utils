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