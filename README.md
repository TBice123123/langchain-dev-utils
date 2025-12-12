# 🦜️🧰 langchain-dev-utils

<p align="center">
    <em>🚀 High-efficiency toolkit designed for LangChain and LangGraph developers</em>
</p>

<p align="center">
  📚 <a href="https://tbice123123.github.io/langchain-dev-utils/">English</a> • 
  <a href="https://tbice123123.github.io/langchain-dev-utils/zh/">中文</a>
</p>

[![PyPI](https://img.shields.io/pypi/v/langchain-dev-utils.svg?color=%2334D058&label=pypi%20package)](https://pypi.org/project/langchain-dev-utils/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11|3.12|3.13|3.14-%2334D058)](https://www.python.org/downloads)
[![Downloads](https://static.pepy.tech/badge/langchain-dev-utils/month)](https://pepy.tech/project/langchain-dev-utils)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://tbice123123.github.io/langchain-dev-utils)

> This is the English version. For the Chinese version, please visit [中文版本](https://github.com/TBice123123/langchain-dev-utils/blob/master/README_cn.md)

## ✨ Why choose langchain-dev-utils?

Tired of writing repetitive code in LangChain development? `langchain-dev-utils` is the solution you need! This lightweight yet powerful toolkit is designed to enhance the development experience of LangChain and LangGraph, helping you:

- ⚡ **Boost development efficiency** - Reduce boilerplate code, allowing you to focus on core functionality
- 🧩 **Simplify complex workflows** - Easily manage multi-model, multi-tool, and multi-agent applications
- 🔧 **Enhance code quality** - Improve consistency and readability, reducing maintenance costs
- 🎯 **Accelerate prototype development** - Quickly implement ideas, iterate and validate faster


## 🎯 Core Features

- **🔌 Unified model management** - Specify model providers through strings, easily switch and combine different models
- **💬 Flexible message handling** - Support for chain-of-thought concatenation, streaming processing, and message formatting
- **🛠️ Powerful tool calling** - Built-in tool call detection, parameter parsing, and human review functionality
- **🤖 Efficient Agent development** - Simplify agent creation process, expand more common middleware
- **📊 Flexible state graph composition** - Support for serial and parallel composition of multiple StateGraphs

## ⚡ Quick Start

1. Install `langchain-dev-utils`:

   ```bash
   pip install -U langchain[openai] langchain-dev-utils
   ```

2. Start using:

   ```python
    from langchain.tools import tool
    from langchain_core.messages import HumanMessage
    from langchain_dev_utils.chat_models import register_model_provider, load_chat_model
    from langchain_dev_utils.agents import create_agent

    # Register model provider
    register_model_provider("vllm", "openai-compatible", "http://localhost:8000/v1")

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

**For more features of this library, please visit the [full documentation](https://tbice123123.github.io/langchain-dev-utils/)**


## 🛠️ GitHub Repository

Visit the [GitHub repository](https://github.com/TBice123123/langchain-dev-utils) to view the source code and report issues.

---

<div align="center">
  <p>Developed with ❤️ and ☕</p>
  <p>If this project helps you, please give us a ⭐️</p>
</div>