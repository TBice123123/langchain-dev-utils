# 🦜️🧰 langchain-dev-utils

<p align="center">
    <em>A utility library for LangChain and LangGraph development.</em>
</p>

[![PyPI](https://img.shields.io/pypi/v/langchain-dev-utils.svg?color=%2334D058&label=pypi%20package)](https://pypi.org/project/langchain-dev-utils/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11|3.12|3.13|3.14-%2334D058)](https://www.python.org/downloads)
[![Downloads](https://static.pepy.tech/badge/langchain-dev-utils/month)](https://pepy.tech/project/langchain-dev-utils)

When building complex large language model applications with LangChain and LangGraph, the development process is not always efficient, and developers often need to write a lot of boilerplate code for regular functionality. To help developers focus more on writing core business logic, `langchain-dev-utils` was created.

This is a lightweight but practical utility library focused on improving the development experience of LangChain and LangGraph. It provides a series of ready-to-use practical utility functions, thereby reducing repetitive code and enhancing code consistency and readability. By simplifying the development path, `langchain-dev-utils` enables you to implement functional prototypes faster, iterate more smoothly, and helps build clearer and more reliable AI large model applications.

## Use Cases

- **Regular Large Language Model Applications**

`langchain-dev-utils` provides a series of ready-to-use tools that can significantly improve the development efficiency of large language model applications. For example, its model management module allows developers to directly specify model providers using strings, which is particularly suitable for scenarios that require dynamically specifying models or need to integrate models from multiple different providers.

- **Complex Agent Development**

`langchain-dev-utils` provides deep optimization support for complex agent applications. This toolset not only offers richer agent middleware but also further encapsulates the tool calling process. Additionally, it specifically provides two efficient pipeline utility functions to facilitate the orchestration and combination of multiple independent agents.

## Key Features

- **Unified Model Management Mechanism**: Simplify model invocation and switching through centralized registration and management of Chat and Embeddings models, improving development efficiency.
- **More Flexible Message Processing**: Provide rich Message class utility functions, supporting chain-of-thought concatenation, streaming chunk merging, message formatting, etc., facilitating the construction of complex dialogue logic.
- **More Powerful Tool Calling Support**: Built-in tool calling detection, parameter parsing, and human review intervention capabilities, enhancing the security and controllability of Agent interactions with external tools.
- **More Efficient Agent Development**: Encapsulate the official Agent creation process, integrate commonly used middleware, and accelerate the construction and iteration of intelligent agents.
- **More Flexible StateGraph Composition**: Support combining multiple StateGraphs in serial or parallel ways, enabling visualization and modular orchestration of complex workflows.

## GitHub Repository

Visit the [GitHub Repository](https://github.com/TBice123123/langchain-dev-utils) to view source code and issues.