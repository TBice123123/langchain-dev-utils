# 概述

中间件是专为 LangChain 预构建 Agent 设计的可插拔组件，旨在实现对 Agent 内部行为的精细化控制。除 LangChain 框架内置的中间件外，本库结合实际应用场景，进一步补充了更为丰富的中间件支持。


本库提供的中间件包括：

- [`PlanMiddleware`](plan.md)：任务规划，将复杂任务拆解为有序子任务
- [`ModelRouterMiddleware`](router.md)：根据输入内容动态路由到最适配的模型
- [`HandoffAgentMiddleware`](handoffs.md)：在多个子 Agent 之间灵活交接任务
- [`ToolCallRepairMiddleware`](tool-call-repair.md)：自动修复大模型无效工具调用
- [`FormatPromptMiddleware`](format.md)：动态格式化系统提示词中的占位符

此外，本库还扩充了官方中间件的功能，增强了模型配置的可用性，支持通过字符串参数指定模型：

- SummarizationMiddleware
- LLMToolSelectorMiddleware
- ModelFallbackMiddleware
- LLMToolEmulator


!!! note "注意"

    后续示例中，我们均是从 `langchain_dev_utils.agents` 中导入了 `create_agent` 函数，而不是 `langchain.agents`。这是因为本库也提供了一个与官方 `create_agent` 函数功能完全相同的函数，只是扩充了通过字符串指定模型的功能。这使得可以直接使用 `register_model_provider` 注册的模型，而无需初始化模型实例后传入。

    示例运行前，请确保：

    1.注册 `vllm` 模型提供商

    ```python
    register_model_provider(
        "vllm",
        "openai-compatible",
        base_url="http://localhost:8000/v1",
    )
    ```

    2.从 `langchain_dev_utils.agents` 中导入 `create_agent` 函数

    ```python
    from langchain_dev_utils.agents import create_agent
    ```
