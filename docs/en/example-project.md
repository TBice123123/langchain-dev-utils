# LangChain Development Utilities Example Project

## Project Overview

This [example project](https://github.com/TBice123123/langchain-dev-utils-example) demonstrates how to efficiently build two typical types of agent systems using the tools and abstractions provided by `langchain-dev-utils`:

- **Simple Agent**: Suitable for end-to-end automation of simple tasks.

- **Supervisor-Multi-Agent Architecture**: Coordinates multiple specialized agents through a supervisor, suitable for complex scenarios that require division of labor, planning, and iteration.

## Project Structure

```
langchain-dev-utils-example/
├── lib/                          # Library modules
│   ├── __init__.py               # Library initialization
│   ├── models.py                 # Model definitions and tools
│   └── register.py               # Model provider registration
├── src/                          # Source code
│   └── agents/                   # Agent implementations
│       ├── __init__.py           # Agent module initialization
│       ├── simple_agent/         # Simple agent example
│       │   ├── __init__.py       # Simple agent module initialization
│       │   ├── agent.py          # Simple agent definition
│       │   └── context.py        # Context schema definition
│       └── supervisor/           # Supervisor multi-agent architecture
│           ├── __init__.py       # Supervisor module initialization
│           ├── supervisor.py     # Supervisor agent
│           └── subagent.py       # Specialized sub-agents
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore file
├── .python-version               # Python version specification
├── LICENSE                       # MIT License
├── README.md                     # English documentation
├── README_cn.md                  # Chinese documentation
├── langgraph.json               # LangGraph configuration
├── pyproject.toml               # Project dependencies
└── uv.lock                      # Dependency lock file
```

## Environment Setup

1. Clone this repository:
```bash
git clone https://github.com/TBice123123/langchain-dev-utils-example.git
cd langchain-dev-utils-example
```

2. Install dependencies using uv:
```bash
uv sync --all-groups
```

3. Configure environment variables:
```bash
cp .env.example .env
```
Edit the `.env` file and add your API key:
```
OPENROUTER_API_KEY=your_openrouter_api_key
```

4. Run the project:
```bash
langgraph dev
```

!!! success "Project Best Practice"
    In this example, we have extended the parameter list for the `load_chat_model` function in the `lib/models.py` module.
    Since `load_chat_model` (and the official `init_chat_model`) needs to support multiple model providers (like vLLM, OpenRouter, etc.), and the initialization parameters for conversational models vary across providers, this library uniformly adopts a **keyword argument (kwargs)** approach for passing model configurations.
    While this approach enhances universality and flexibility, it weakens the IDE's type hinting capabilities, potentially increasing the risk of parameter typos or type misuse.

    Therefore, it is recommended that in actual projects, once the model provider is determined, you can explicitly extend the parameter signature for its corresponding chat model integration class to restore full type hinting and improve the development experience.

    Additionally, in `lib/register.py`, we have encapsulated the `register_model_provider` registration logic into a function, which is imported in `lib/__init__.py`. As long as the `lib` module is imported, the model provider registration is completed.