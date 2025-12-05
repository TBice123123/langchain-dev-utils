# What is langchain-dev-utils?

Building complex large language model applications with LangChain and LangGraph isn't always efficient, as developers often need to write substantial boilerplate code for common functionalities. To help developers focus more on core business logic, `langchain-dev-utils` was created.

This is a lightweight yet practical utility library focused on enhancing the development experience with LangChain and LangGraph. It provides a series of out-of-the-box utility functions, thereby reducing repetitive code and improving code consistency and readability. By streamlining the development process, `langchain-dev-utils` enables you to prototype features faster and iterate more smoothly, helping you build clearer and more reliable AI-powered applications.

## Use Cases

- **General LLM Applications**

`langchain-dev-utils` offers a range of ready-to-use tools that can significantly boost development efficiency for LLM applications. For example, its model management module allows developers to specify model providers directly using strings, which is particularly useful for scenarios requiring dynamic model specification or integration of models from multiple different providers.

- **Complex Agent Development**

`langchain-dev-utils` provides deeply optimized support for complex agent applications. The toolkit not only includes richer agent middleware but also further encapsulates the tool calling process. Additionally, it specifically offers two efficient pipeline utility functions to facilitate the orchestration and composition of multiple independent agents.

## Documentation

`langchain-dev-utils` contains the following five modules. The first four are primarily used with `langchain`, while the last is mainly for `langgraph`.

- **[Model Management](./model-management/chat.md)** - Provides a unified registration and management approach, making it convenient for developers to use chat and embedding models.
- **[Message Conversion](./message-conversion/message.md)** - Provides utility functions related to the Message class, including chain-of-thought concatenation, chunk merging, list string formatting, and more.
- **[Tool Calling](./tool-calling/tool.md)** - Provides utility functions related to tool calling, including tool call detection and parameter parsing, and adding human review to tool calls.
- **[Agent Development](./agent-development/multi-agent.md)** - Provides utility functions to facilitate Agent development, including further encapsulation of official Agent creation functions and common middleware.
- **[State Graph Orchestration Pipeline](./graph-orchestration/pipeline.md)** - Provides pipeline utility functions to combine multiple state subgraphs (StateGraph) in parallel or serial.

## Key Features

- **Unified Model Management Mechanism**: Simplifies model invocation and switching through centralized registration and management of Chat and Embeddings models, enhancing development efficiency.
- **More Flexible Message Handling**: Offers rich Message class utility functions, supporting chain-of-thought concatenation, streaming chunk merging, message formatting, etc., facilitating the construction of complex dialogue logic.
- **More Powerful Tool Calling Support**: Built-in capabilities for tool call detection, parameter parsing, and human review intervention, enhancing the security and controllability of Agent interactions with external tools.
- **More Efficient Agent Development**: Encapsulates the official Agent creation process and integrates common middleware, accelerating the construction and iteration of agents.
- **More Flexible State Graph Composition**: Supports combining multiple StateGraphs in serial or parallel, enabling the visualization and modular orchestration of complex workflows.

## GitHub Repository

Visit the [GitHub repository](https://github.com/TBice123123/langchain-dev-utils) to view the source code and issues.
