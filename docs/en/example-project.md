# Langchain-dev-utils Example Project

## Project Overview

This [example project](https://github.com/TBice123123/langchain-dev-utils-example) demonstrates how to efficiently build two typical types of agent systems using the tools and abstractions provided by `langchain-dev-utils`:

- **Simple Agent**: Suitable for end-to-end automation of simple tasks.

- **Supervisor-Multi-Agent Architecture**: A supervisor coordinates multiple specialized agents, suitable for complex scenarios that require division of labor, planning, and iteration.

## Project Structure

```
langchain-dev-utils-example/
├── src/                          
│   ├── agents/                   
│   │   ├── __init__.py           
│   │   ├── simple_agent/        
│   │   │   ├── __init__.py       
│   │   │   ├── agent.py         
│   │   │   └── context.py        
│   │   └── supervisor/           
│   │       ├── __init__.py       
│   │       ├── supervisor.py    
│   │       └── subagent.py      
│   └── utils/                    
│       ├── __init__.py           
│       ├── models.py             
│       └── register.py          
├── .env.example                  
├── .gitignore                    
├── .python-version               
├── LICENSE                       
├── README.md                     
├── README_cn.md                  
├── langgraph.json               
├── pyproject.toml               
└── uv.lock                    
```

## Installation Steps

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
Edit the `.env` file and add your API keys (for `OpenRouter` and `Tavily`):
```
OPENROUTER_API_KEY=your_openrouter_api_key
TAVILY_API_KEY=your_tavily_api_key
```

4. Run LangGraph CLI

The project is configured to be compatible with the LangGraph CLI. Use the following command to start the agent:

```bash
langgraph dev
```

!!! success "Project Best Practice"
    In this example, we have extended the parameter list of the `load_chat_model` function in the `src/utils/models.py` module.
    Since `load_chat_model` (and the official `init_chat_model`) needs to support multiple model providers (like vLLM, OpenRouter, etc.), and the initialization parameters for chat models vary across providers, this library uniformly adopts the **keyword arguments (kwargs)** approach for passing model configurations.
    While this approach enhances universality and flexibility, it weakens the IDE's type hinting capabilities, which may increase the risk of parameter typos or type misuse.

    Therefore, it is recommended that in actual projects, if the model provider is already determined, you can explicitly extend the parameter signature for its corresponding chat model integration class to restore full type hinting and improve the development experience.

    Additionally, in `src/utils/register.py`, we have encapsulated the logic for `register_model_provider` into a function, which is imported in `src/utils/__init__.py`. As long as the `src/utils` module is imported, the model provider registration is completed.