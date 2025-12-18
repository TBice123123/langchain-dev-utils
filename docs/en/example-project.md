# Langchain-dev-utils Example Project

This repository provides an example project [`langchain-dev-utils-example`](https://github.com/TBice123123/langchain-dev-utils-example) that demonstrates how to efficiently build two typical agent systems using the utility functions provided by `langchain-dev-utils`:

- **Single Agent**: Suitable for executing simple tasks and tasks related to long-term memory storage.
- **Supervisor-Multi-Agent Architecture**: Coordinates multiple specialized agents through a central supervisor, suitable for complex scenarios requiring task decomposition, planning, and iterative optimization.

<p align="center">
  <img src="../../assets/graph.png" alt="graph">
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
4. Edit the `.env` file and fill in your API keys (requires API keys for `OpenRouter` and `Tavily`)

5. Start the project
```bash
langgraph dev
```

## Features Used

**Single Agent**:

Features from this library used:

- Chat model management: `register_model_provider`, `load_chat_model`
- Embedding model management: `register_embeddings_provider`, `load_embedding_model`
- Sequence formatting: `format_sequence`
- Middleware: `format_prompt`

**Supervisor-Multi-Agent Architecture**:

Features from this library used:

- Chat model management: `register_model_provider`, `load_chat_model`
- Multi-agent construction: `wrap_agent_as_tool`

!!! success "Best Practices for Model Management Features"
    In this example, the following best practices are adopted for model management features (chat model management, embedding model management):

    In the example project's `src/utils/providers/chat_models/load.py` and `src/utils/providers/embeddings/load.py`, the parameter lists of the `load_chat_model` and `load_embeddings` functions are extended. Since these functions need to support multiple model providers (such as vLLM, OpenRouter, etc.), and each provider's model initialization parameters are different, this library uniformly uses **keyword arguments (kwargs)** to pass additional model parameters (LangChain's official functions `init_chat_model` and `init_embeddings` also adopt this approach). While this approach improves universality and flexibility, it weakens IDE type hinting capabilities and may increase the risk of parameter spelling errors or type misuse.

    Therefore, it is recommended that in actual projects, if the model provider to be used has been determined, you can extend the parameter signatures for its corresponding model integration classes **as needed** to restore complete type hints and development experience.
    
    At the same time, this library requires `register_model_provider` and `register_embeddings_provider` to be executed at project startup. Therefore, this project encapsulates the related registration logic functions in `src/utils/providers/chat_models/register.py` and `src/utils/providers/embeddings/register.py`, and automatically executes registration in their respective `__init__.py`; as long as the `src/utils/providers` module is imported, the registration of model providers can be completed.