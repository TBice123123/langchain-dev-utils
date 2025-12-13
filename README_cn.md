# 🦜️🧰 langchain-dev-utils

<p align="center">
    <em>🚀 专为 LangChain 和 LangGraph 开发者打造的高效工具库</em>
</p>

<p align="center">
  📚 <a href="https://tbice123123.github.io/langchain-dev-utils/">English</a> • 
  <a href="https://tbice123123.github.io/langchain-dev-utils/zh/">中文</a>
</p>

[![PyPI](https://img.shields.io/pypi/v/langchain-dev-utils.svg?color=%2334D058&label=pypi%20package)](https://pypi.org/project/langchain-dev-utils/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11|3.12|3.13|3.14-%2334D058)](https://www.python.org/downloads)
[![Downloads](https://static.pepy.tech/badge/langchain-dev-utils/month)](https://pepy.tech/project/langchain-dev-utils)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://tbice123123.github.io/langchain-dev-utils/zh/)

> 当前为中文版，英文版请访问[English Version](https://github.com/TBice123123/langchain-dev-utils/blob/master/README.md)

## ✨ 为什么选择 langchain-dev-utils？

厌倦了在 LangChain 开发中编写重复代码？`langchain-dev-utils` 正是您需要的解决方案！这个轻量但功能强大的工具库专为提升 LangChain 和 LangGraph 开发体验而设计，帮助您：

- ⚡ **提升开发效率** - 减少样板代码，让您专注于核心功能
- 🧩 **简化复杂流程** - 轻松管理多模型、多工具和多智能体应用
- 🔧 **增强代码质量** - 提高一致性和可读性，减少维护成本
- 🎯 **加速原型开发** - 快速实现想法，更快迭代验证


## 🎯 核心功能

- **🔌 统一的模型管理** - 通过字符串指定模型提供商，轻松切换和组合不同模型
- **💬 灵活的消息处理** - 支持思维链拼接、流式处理和消息格式化
- **🛠️ 强大的工具调用** - 内置工具调用检测、参数解析和人工审核功能
- **🤖 高效的 Agent 开发** - 简化智能体创建流程，扩充更多的常用中间件
- **📊 灵活的状态图组合** - 支持串行和并行方式组合多个 StateGraph

## ⚡ 快速开始

**1. 安装 `langchain-dev-utils`**

```bash
pip install -U "langchain-dev-utils[standard]"
```

**2. 开始使用**

```python
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_dev_utils.chat_models import register_model_provider, load_chat_model
from langchain_dev_utils.agents import create_agent

# 注册模型提供商
register_model_provider("vllm", "openai-compatible", base_url="http://localhost:8000/v1")

@tool
def get_current_weather(location: str) -> str:
    """获取指定地点的当前天气"""
    return f"25度，{location}"

# 使用字符串动态加载模型
model = load_chat_model("vllm:qwen3-4b")
response = model.invoke("你好")
print(response)

# 创建智能体
agent = create_agent("vllm:qwen3-4b", tools=[get_current_weather])
response = agent.invoke({"messages": [HumanMessage(content="今天纽约的天气如何？")]})
print(response)
```

**获取更多的本库功能，请访问[完整文档](https://tbice123123.github.io/langchain-dev-utils/zh/)**


## 🛠️ GitHub 仓库

访问 [GitHub 仓库](https://github.com/TBice123123/langchain-dev-utils) 查看源代码和问题。


---

<div align="center">
  <p>由 ❤️ 和 ☕ 驱动开发</p>
  <p>如果这个项目对您有帮助，请给我们一个 ⭐️</p>
</div>