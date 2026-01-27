# 🦜️🧰 langchain-dev-utils

<p align="center">
    <em>🚀 An efficient toolkit designed specifically for LangChain and LangGraph developers</em>
</p>


[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-black.svg?logo=github)](https://github.com/TBice123123/langchain-dev-utils)
[![PyPI](https://img.shields.io/pypi/v/langchain-dev-utils.svg?color=%2334D058&label=pypi%20package&logo=python)](https://pypi.org/project/langchain-dev-utils/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg?logo=python&label=Python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?label=License)](https://opensource.org/licenses/MIT)
[![Last Commit](https://img.shields.io/github/last-commit/TBice123123/langchain-dev-utils)](https://github.com/TBice123123/langchain-dev-utils)


## Why choose langchain-dev-utils?

Tired of writing repetitive code in LangChain development? `langchain-dev-utils` is exactly the solution you need! This lightweight yet powerful toolkit is designed to enhance the development experience of LangChain and LangGraph, helping you:

- **Boost development efficiency** - Reduce boilerplate code, allowing you to focus on core functionality
- **Simplify complex workflows** - Easily manage multi-model, multi-tool, and multi-agent applications
- **Enhance code quality** - Improve consistency and readability, reduce maintenance costs
- **Accelerate prototype development** - Quickly implement ideas, iterate and validate faster


## Core Features

<div class="grid cards" markdown>

-   :fontawesome-solid-chain: __Unified Model Management__
    
    ---
    
    Specify model providers through strings, easily switch and combine different models.

-    :fontawesome-solid-file-code: __Built-in OpenAI-Compatible Integration Class__
    
    ---
    
    Built-in OpenAI-Compatible API integration class, improving model compatibility through explicit configuration.
    
-   :material-message-fast: __Flexible Message Processing__
    
    ---
    
    Supports chain-of-thought concatenation, streaming processing, and message formatting
    
-   :material-tools: __Powerful Tool Calling__
    
    ---
    
    Built-in tool calling detection, parameter parsing, and human review functions
    
-   :octicons-dependabot-16: __Efficient Agent Development__
    
    ---
    
    Simplifies the agent creation process and expands more common middleware
    
-   :fontawesome-solid-circle-nodes: __Convenient State Graph Building__
    
    ---
    
    Provides pre-built two functions for easily constructing sequential and parallel state graphs.

</div>


## Quick Start

**1. Install `langchain-dev-utils`**

```bash
pip install -U "langchain-dev-utils[standard]"
```

**2. Get Started**

```python
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_dev_utils.chat_models import register_model_provider, load_chat_model
from langchain_dev_utils.agents import create_agent

# Register model provider
register_model_provider("vllm", "openai-compatible", base_url="http://localhost:8000/v1")

@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for the specified location"""
    return f"25 degrees, {location}"

# Dynamically load model using string
model = load_chat_model("vllm:qwen3-4b")
response = model.invoke("Hello")
print(response)

# Create agent
agent = create_agent("vllm:qwen3-4b", tools=[get_current_weather])
response = agent.invoke({"messages": [HumanMessage(content="What's the weather like in New York today?")]})
print(response)
```


## GitHub Repository

Visit the [GitHub Repository](https://github.com/TBice123123/langchain-dev-utils) to view source code and issues.