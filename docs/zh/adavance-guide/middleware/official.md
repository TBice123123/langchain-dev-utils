### LangChain 内置中间件扩展

本库对以下官方中间件进行了增强，支持通过字符串直接指定模型——**前提是模型必须已通过 `register_model_provider` 注册**：

- `SummarizationMiddleware`
- `LLMToolSelectorMiddleware`
- `ModelFallbackMiddleware`
- `LLMToolEmulator`

导入本库中的中间件即可使用，用法与官方一致：

```python
from langchain_core.messages import AIMessage
from langchain_dev_utils.agents.middleware import SummarizationMiddleware
from langchain_dev_utils.chat_models import register_model_provider

# 必须先通过 register_model_provider 注册模型
register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)

agent = create_agent(
    model="vllm:qwen3-4b",
    middleware=[
        SummarizationMiddleware(
            model="vllm:qwen3-4b",
            trigger=("tokens", 50),
            keep=("messages", 1),
        )
    ],
    system_prompt="你是一个智能的AI助手，可以解决用户的问题",
)
response = agent.invoke({"messages": messages})
print(response)
```