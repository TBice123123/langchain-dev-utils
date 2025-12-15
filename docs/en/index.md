# 🦜️🧰 langchain-dev-utils

<p align="center">
    <em>🚀 A high-efficiency toolkit designed specifically for LangChain and LangGraph developers</em>
</p>

[![PyPI](https://img.shields.io/pypi/v/langchain-dev-utils.svg?color=%2334D058&label=pypi%20package)](https://pypi.org/project/langchain-dev-utils/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11|3.12|3.13|3.14-%2334D058)](https://www.python.org/downloads)
[![Downloads](https://static.pepy.tech/badge/langchain-dev-utils/month)](https://pepy.tech/project/langchain-dev-utils)

## Why choose langchain-dev-utils?

Tired of writing repetitive code in LangChain development? `langchain-dev-utils` is exactly the solution you need! This lightweight yet powerful toolkit is designed specifically to enhance the development experience of LangChain and LangGraph, helping you:

- **Improve development efficiency** - Reduce boilerplate code, allowing you to focus on core functionality
- **Simplify complex processes** - Easily manage multi-model, multi-tool, and multi-agent applications
- **Enhance code quality** - Improve consistency and readability, reducing maintenance costs
- **Accelerate prototype development** - Quickly implement ideas, iterate and validate faster

## Core Features

<div class="grid cards" markdown>

-   __Unified Model Management__
    
    ---
    
    Easily switch and combine different models by specifying model providers through strings
    
-   __Flexible Message Processing__
    
    ---
    
    Support chain-of-thought concatenation, streaming processing, and message formatting
    
-   __Powerful Tool Invocation__
    
    ---
    
    Built-in tool invocation detection, parameter parsing, and manual review features
    
-   __Efficient Agent Development__
    
    ---
    
    Simplify the agent creation process and expand more common middleware
    
-   __Flexible State Graph Composition__
    
    ---
    
    Support serial and parallel composition of multiple StateGraphs

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