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
4. 编辑 `.env` 文件，填入你的 API 密钥（需要 `ZhipuAI` 和 `Tavily` 的 API 密钥）。
  
5. 启动项目
```bash
langgraph dev
```

## 使用的功能

**单智能体（Simple Agent）**：

使用的本库的功能：

- 对话模型管理（含OpenAI兼容API集成）：`register_model_provider`、`load_chat_model`
- 嵌入模型管理：`register_embeddings_provider`、`load_embeddings`
- 格式化序列：`format_sequence`
- 中间件：`format_prompt`

**监督者-多智能体架构（Supervisor-Multi-Agent Architecture）**：

使用的本库的功能：

- 对话模型管理（含OpenAI兼容API集成）：`register_model_provider`、`load_chat_model`
- 多智能体构建：`wrap_agent_as_tool`


## 如何自定义

可以根据实际的需求，对本项目进行自定义修改。

### 1. 替换对话模型提供商

本项目默认使用智谱AI的GLM系列作为核心模型，具体如下：

  - `GLM-4.7`：用于`simple-agent`
  - `GLM-4.6`：用于`supervisor-agent`的`supervisor`
  - `GLM-4.5`：用于`supervisor-agent`的`subagent`
  - `GLM-4.6V`：用于`supervisor-agent`的`vision subagent`

如需自定义模型提供商，请修改`src/utils/providers/chat_models/register.py`，在`register_all_model_providers`函数中使用`register_model_provider`函数注册你的模型提供商。

同时建议修改`src/utils/providers/chat_models/load.py`，在`load_chat_model`函数中添加对应的加载逻辑。

!!! success "对话模型管理最佳实践"
    `load_chat_model`函数采用关键字参数接收不同对话模型类的额外参数（LangChain官方函数也采用此方式）。这种方式提升了通用性，但会削弱IDE类型提示，增加参数误用风险。因此，若已确定具体提供商，可针对其集成对话模型类（或嵌入模型类）扩展参数签名以恢复类型提示，参考`src/utils/providers/chat_models/load.py`进行针对性修改。

### 2. 注册嵌入模型提供商

嵌入模型提供商的注册方式与对话模型类似。请修改`src/utils/providers/embeddings/register.py`，在`register_all_embeddings_providers`函数中使用`register_embeddings_provider`函数注册你的嵌入模型提供商。

如需自定义加载逻辑，可修改`src/utils/providers/embeddings/load.py`，在`load_embeddings`函数中添加相应的加载逻辑。


### 3. 自定义工具

**单智能体（simple-agent）**  
工具实现位于 `src/agents/simple_agent/tools.py`，已内置：  
- `save_user_memory` —— 持久化用户记忆  
- `get_user_memory` —— 读取用户记忆  

如需扩展，直接在该文件内新增对应的工具实现即可。

**监督者-多智能体（supervisor-agent）**  
工具实现位于 `src/agents/supervisor/subagent/tools.py`。是子智能体的工具实现，如需为子智能体添加自定义工具，直接在该文件内新增对应的工具实现即可。
  
注意：`supervisor` 默认仅持有“调用子智能体”的两个工具。若需为 `supervisor` 追加自定义工具，建议在 `src/agents/supervisor/` 下新建 `tools.py`，编写完成后在 `src/agents/supervisor/agent.py` 中导入并传递给 `create_agent` 函数即可。
