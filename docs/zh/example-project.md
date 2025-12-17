# Langchain-dev-utils 示例项目

## 项目概述

本[示例项目](https://github.com/TBice123123/langchain-dev-utils-example)演示了如何利用 `langchain-dev-utils` 提供的工具与抽象，高效构建两类典型智能体系统：

- **单智能体（Simple Agent）**：适用于端到端的简单任务自动化。

- **主管-多智能体架构（Supervisor-Multi-Agent）**：通过一个主管协调多个专业智能体，适用于需分工、规划与迭代的复杂场景。

## 项目结构

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

## 安装步骤

1. 克隆此仓库：
```bash
git clone https://github.com/TBice123123/langchain-dev-utils-example.git
cd langchain-dev-utils-example
```

2. 使用 uv 安装依赖：
```bash
uv sync --all-groups
```

3. 配置环境变量：
```bash
cp .env.example .env
```
编辑 `.env` 文件，添加您的 API 密钥 (需要`OpenRouter`和`Tavily`)
```
OPENROUTER_API_KEY=your_openrouter_api_key
TAVILY_API_KEY=your_tavily_api_key
```
4. 运行 LangGraph CLI
```bash
langgraph dev
```

!!! success "项目最佳实践"
    在本示例中，我们在 `src/utils/models.py` 模块中对 `load_chat_model` 函数的参数列表进行了扩展处理。
    由于 `load_chat_model`（以及官方函数 `init_chat_model`）需要支持多种模型提供商（如 vLLM、OpenRouter 等），而各提供商的对话模型初始化参数各不相同，因此本库统一采用**关键字参数（kwargs)** 的方式传入模型配置。
    这种方式虽提高了通用性和灵活性，但会削弱 IDE 的类型提示能力，可能增加参数拼写错误或类型误用的风险。

    故推荐在实际项目中，若已确定所使用的模型提供商，可针对其对应的对话模型集成类**按需**扩展参数签名，以恢复完整的类型提示和开发体验。

    同时在 `src/utils/register.py` 中封装了 `register_model_provider` 注册逻辑的函数，并在 `src/utils/__init__.py` 中导入；只要引入 `src/utils` 模块，即可完成模型提供方的注册。

