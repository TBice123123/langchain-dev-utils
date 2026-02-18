# 🦜️🧰 langchain-dev-utils

<p align="center">
    <em>🚀 An efficient utility library crafted for LangChain and LangGraph developers</em>
</p>

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-black.svg?logo=github)](https://github.com/TBice123123/langchain-dev-utils)
[![PyPI](https://img.shields.io/pypi/v/langchain-dev-utils.svg?color=%2334D058&label=pypi%20package&logo=python)](https://pypi.org/project/langchain-dev-utils/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg?logo=python&label=Python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?label=License)](https://opensource.org/licenses/MIT)
[![Last Commit](https://img.shields.io/github/last-commit/TBice123123/langchain-dev-utils)](https://github.com/TBice123123/langchain-dev-utils)


## Why choose langchain-dev-utils?

Tired of writing repetitive code in LangChain development? `langchain-dev-utils` is exactly the solution you need! This lightweight yet powerful utility library is designed to enhance the LangChain and LangGraph development experience, helping you to:

- **Boost Development Efficiency** - Reduce boilerplate code, allowing you to focus on core functionality.
- **Simplify Complex Workflows** - Easily manage multi-model, multi-tool, and multi-agent applications.
- **Enhance Code Quality** - Improve consistency and readability, reducing maintenance costs.
- **Accelerate Prototyping** - Quickly implement ideas and iterate faster for validation.


## Core Features

<div class="grid cards" markdown>

-   :fontawesome-solid-chain: __Unified Model Management__
    
    ---
    
    Easily switch and combine different models by specifying the model provider via string.

-    :fontawesome-solid-file-code: __Built-in OpenAI-Compatible Integration__
    
    ---
    
    Built-in integration classes for OpenAI-Compatible APIs, enhancing model compatibility through explicit configuration.
    
-   :material-message-fast: __Flexible Message Handling__
    
    ---
    
    Supports chain-of-thought concatenation, streaming processing, and message formatting.
    
-   :material-tools: __Powerful Tool Calling__
    
    ---
    
    Built-in tool call detection, parameter parsing, and human-in-the-loop review functionality.
    
-   :octicons-dependabot-16: __Efficient Agent Development__
    
    ---
    
    Simplifies the agent creation process and provides more common middleware extensions.
    
-   :fontawesome-solid-circle-nodes: __Convenient State Graph Construction__
    
    ---
    
    Provides two pre-built functions to facilitate the construction of sequential and parallel execution state graphs.

</div>

## Quick Start

One of the main uses of this library is integrating models that provide OpenAI-Compatible APIs. Below is an example using `qwen2.5-7b` deployed via vLLM.

**1. Install `langchain-dev-utils`**

```bash
pip install -U "langchain-dev-utils[standard]"
```

**Note**: You must install the `langchain-dev-utils[standard]` version to use this feature.

**2. Getting Started**

```python
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_dev_utils.chat_models import register_model_provider, load_chat_model
from langchain_dev_utils.agents import create_agent

# Register model provider
register_model_provider("vllm", "openai-compatible", base_url="http://localhost:8000/v1")

@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a specified location."""
    return f"25 degrees, {location}"

# Dynamically load model using string
model = load_chat_model("vllm:qwen2.5-7b")
response = model.invoke("Hello")
print(response)

# Create agent
agent = create_agent("vllm:qwen2.5-7b", tools=[get_current_weather])
response = agent.invoke({"messages": [HumanMessage(content="What is the weather in New York today?")]})
print(response)
```


## GitHub Repository

Visit the [GitHub Repository](https://github.com/TBice123123/langchain-dev-utils) to view the source code and issues.