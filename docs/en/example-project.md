# Langchain-dev-utils Example Project

This repository provides an example project [`langchain-dev-utils-example`](https://github.com/TBice123123/langchain-dev-utils-example) designed to help developers quickly understand how to efficiently build two typical intelligent agent (agent) systems using the utility functions provided by `langchain-dev-utils`:

- **Single Agent**: Suitable for executing simple tasks and tasks related to long-term memory storage.
- **Supervisor-Multi-Agent Architecture**: Coordinates multiple specialized agents through a central supervisor, suitable for complex scenarios that require task decomposition, planning, and iterative optimization.

<p align="center">
  <img src="../assets/graph.png" alt="graph">
</p>

## Quick Start

1. Clone this repository:
```bash
git clone https://github.com/TBice123123/langchain-dev-utils-example.git  
cd langchain-dev-utils-example
```
2. Install dependencies using uv:
```bash
uv sync
```
3. Create .env file
```bash
cp .env.example .env
```
4. Edit the `.env` file and fill in your API keys (requires API keys from `ZhipuAI` and `Tavily`).
  
5. Start the project
```bash
langgraph dev
```

## Features Used

**Single Agent**:

Features from this library used:

- Chat model management: `register_model_provider`, `load_chat_model`
- Embedding model management: `register_embeddings_provider`, `load_embeddings`
- Format sequence: `format_sequence`
- Middleware: `format_prompt`

**Supervisor-Multi-Agent Architecture**:

Features from this library used:

- Chat model management: `register_model_provider`, `load_chat_model`
- Multi-agent construction: `wrap_agent_as_tool`

## How to Customize

You can customize this project according to your actual needs.

### 1. Replace Chat Model Provider

This project uses ZhipuAI's GLM series as the core model by default, specifically as follows:

  - `GLM-4.7`: Used for `simple-agent`
  - `GLM-4.6`: Used for `supervisor-agent`
  - `GLM-4.5`: Used for the `supervisor` of `supervisor-agent`

If you want to customize your model provider, you need to modify the content in `src/utils/providers/chat_models/register.py`, and register your model provider using the `register_model_provider` function in the `register_all_model_providers` function.

It is also recommended to modify the content in `src/utils/providers/chat_models/load.py`, and add the loading logic for your model provider in the `load_chat_model` function.

!!! success "Chat Model Management Best Practice"
    For additional parameters of different chat model classes, the `load_chat_model` function receives them through keyword arguments (the corresponding LangChain function also uses this method). Although this approach enhances universality, it weakens IDE type hints and increases the risk of parameter misuse. Therefore, if the specific provider is already determined, you can extend the parameter signature for its integrated chat model class (or embedding model class) to restore type hints. You can refer to the content in `src\utils\providers\chat_models\load.py` for targeted modifications.

### 2. Register Embedding Model Provider

Similar to chat model providers, you can also register custom embedding model providers as needed. You need to modify the content in `src/utils/providers/embeddings/register.py`, and register your embedding model provider using the `register_embeddings_provider` function in the `register_all_embeddings_providers` function.

If needed, you can also modify the content in `src/utils/providers/embeddings/load.py`, and add the loading logic for your embedding model provider in the `load_embeddings` function.

### 3. Customize Tools

**Single Agent (simple-agent)**  
Tool implementations are located in `src/agents/simple_agent/tools.py`, with built-in:  
- `save_user_memory` - Persist user memory  
- `get_user_memory` - Read user memory  

For extensions, simply add corresponding tool implementations in this file.

**Supervisor-Multi-Agent (supervisor-agent)**  
Tool implementations are located in `src/agents/supervisor/subagent/tools.py`. These are tool implementations for sub-agents. If you need to add custom tools for sub-agents, simply add corresponding tool implementations in this file.
  
Note: The `supervisor` only holds two tools for "calling sub-agents" by default. If you need to add custom tools for the `supervisor`, it is recommended to create a new `tools.py` under `src/agents/supervisor/`, and after writing, import it in `src/agents/supervisor/agent.py` and pass it to the `create_agent` function.