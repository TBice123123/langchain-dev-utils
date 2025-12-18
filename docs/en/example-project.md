# Langchain-dev-utils Example Project

This repository provides an example project [`langchain-dev-utils-example`](https://github.com/TBice123123/langchain-dev-utils-example) designed to help developers quickly understand how to use the utility functions provided by `langchain-dev-utils` to efficiently build two typical agent systems:

- **Single Agent**: Suitable for executing simple tasks and tasks related to long-term memory storage.
- **Supervisor-Multi-Agent Architecture**: Coordinates multiple specialized agents through a central supervisor, suitable for complex scenarios requiring task decomposition, planning, and iterative optimization.

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
4. Edit the `.env` file and enter your API keys (API keys for `OpenRouter` and `Tavily` are required)

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

!!! success "Best Practices for Model Management Features"
    This example performs the following preprocessing for model management features (chat models/embedding models):

    - 1.**Usage of load_chat_model and load_embeddings functions**

    For additional parameters of different chat model classes (or embedding model classes), the `load_chat_model` and `load_embeddings` functions receive them through keyword arguments (LangChain's corresponding two functions also adopt this approach). Although this approach enhances versatility, it weakens IDE type hints and increases the risk of parameter misuse. Therefore, if the specific provider is already determined, you can extend the parameter signatures for its integrated chat model class (or embedding model class) to restore type hints. You can refer to `src\utils\providers\chat_models\load.py` and `src\utils\providers\embeddings\load.py`.

    - 2.**Usage of register_model_provider and register_embeddings_provider functions** 

    The `register_model_provider` and `register_embeddings_provider` functions need to be executed at startup. For this, you can refer to this project's `src\utils\providers\chat_models\register.py` and `src\utils\providers\embeddings\register.py`. These two files encapsulate the registration logic and are imported in `src/utils/providers/__init__.py`. Users only need to import the `src/utils/providers` module to complete the registration of all providers.