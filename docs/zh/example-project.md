# Langchain-dev-utils Example Project

该仓库提供了一个示例项目[`langchain-dev-utils-example`](https://github.com/TBice123123/langchain-dev-utils-example)，演示了如何利用 `langchain-dev-utils` 提供的工具函数，高效构建两种典型的智能体（agent）系统：

- **单智能体（Single Agent）**：适用于执行简单任务以及长期记忆存储相关的任务。
- **监督者-多智能体架构（Supervisor-Multi-Agent Architecture）**：通过一个中央监督者协调多个专业化智能体，适用于需要任务分解、规划和迭代优化的复杂场景。


<p align="center">
  <img src="../../assets/graph.png" alt="graph">
</p>


## 快速开始

1. 克隆本仓库：
```bash
git clone https://github.com/TBice123123/langchain-dev-utils-example.git  
cd langchain-dev-utils-example
```
2. 使用 uv 安装依赖：
```bash
uv sync
```
3. 创建.env文件
```bash
cp .env.example .env
```
4. 编辑 `.env` 文件，填入你的 API 密钥（需要 `OpenRouter` 和 `Tavily` 的 API 密钥）

5. 启动项目
```bash
langgraph dev
```

## 使用的功能

**单智能体（Simple Agent）**：

使用的本库的功能：

- 对话模型管理：`register_model_provider`、`load_chat_model`
- 嵌入模型管理：`register_embeddings_provider`、`load_embedding_model`
- 格式化序列：`format_sequence`
- 中间件：`format_prompt`

**监督者-多智能体架构（Supervisor-Multi-Agent Architecture）**：

使用的本库的功能：

- 对话模型管理：`register_model_provider`、`load_chat_model`
- 多智能体构建：`wrap_agent_as_tool`



!!! success "模型管理功能的最佳实践"
    在本示例中，针对模型管理（对话模型管理、嵌入模型管理）功能，采用了如下最佳实践：

    在示例项目的`src/utils/providers/chat_models/load.py` 和 `src/utils/providers/embeddings/load.py` 中，对 `load_chat_model` 和 `load_embeddings` 函数的参数列表进行了扩展处理。由于这些函数需要支持多种模型提供商（如 vLLM、OpenRouter 等），而各提供商的模型初始化参数各不相同，因此本库在实现过程中统一采用**关键字参数（kwargs）**的方式传入模型额外参数（langchain官方函数`init_chat_model`、`init_embeddings`也同样采用此方式）。这种方式虽提高了通用性和灵活性，但会削弱 IDE 的类型提示能力，可能增加参数拼写错误或类型误用的风险。

    故推荐在实际项目中，若已确定所使用的模型提供商，可针对其对应的模型集成类**按需**扩展参数签名，以恢复完整的类型提示和开发体验。
    
    同时，本库要求`register_model_provider`和`register_embeddings_provider`在项目启动时就需要执行，故本项目同时在`src/utils/providers/chat_models/register.py` 和 `src/utils/providers/embeddings/register.py` 中封装了相关的注册逻辑的函数，并在各自的 `__init__.py` 中自动执行注册；只要引入 `src/utils/providers` 模块，即可完成模型提供方的注册。
