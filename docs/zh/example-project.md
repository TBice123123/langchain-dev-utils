# Langchain-dev-utils Example Project

该仓库提供了一个示例项目[`langchain-dev-utils-example`](https://github.com/TBice123123/langchain-dev-utils-example)，目的是为了帮助开发者快速了解如何利用 `langchain-dev-utils` 提供的工具函数，高效构建两种典型的智能体（agent）系统：

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
- 嵌入模型管理：`register_embeddings_provider`、`load_embeddings`
- 格式化序列：`format_sequence`
- 中间件：`format_prompt`

**监督者-多智能体架构（Supervisor-Multi-Agent Architecture）**：

使用的本库的功能：

- 对话模型管理：`register_model_provider`、`load_chat_model`
- 多智能体构建：`wrap_agent_as_tool`



!!! success "模型管理功能的最佳实践"
    本示例针对模型管理功能（对话模型/嵌入模型）进行了如下的预处理：

    - 1.**load_chat_model和load_embeddings函数使用**

    对于不同对话模型类（或嵌入模型类）的额外参数，`load_chat_model`和`load_embeddings`函数采用关键字参数方式进行接收（LangChain对应的两个函数也采用此方式）。虽然此方式提升了通用性，但会削弱IDE类型提示，增加参数误用风险。因此，若已确定具体提供商，可以针对其集成对话模型类（或嵌入模型类）扩展参数签名以恢复类型提示，可以参考`src\utils\providers\chat_models\load.py`以及`src\utils\providers\embeddings\load.py`。

    - 2.**register_model_provider和register_embeddings_provider函数使用** 

    `register_model_provider`和`register_embeddings_provider`函数需要在启动时完成执行，对此，可参考本项目的`src\utils\providers\chat_models\register.py`以及`src\utils\providers\embeddings\register.py`。这两个文件封装了注册逻辑，并在`src/utils/providers/__init__.py`中进行了导入，使用者只需引入`src/utils/providers`模块，即可完成所有提供商的注册。
