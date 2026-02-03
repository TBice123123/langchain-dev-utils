### LangChain Built-in Middleware Extensions

This library enhances the following official middleware components, supporting direct model specification via strings—**provided the model has been registered via `register_model_provider`**:

- `SummarizationMiddleware`
- `LLMToolSelectorMiddleware`
- `ModelFallbackMiddleware`
- `LLMToolEmulator`

Simply import the middleware from this library to use them, with the same usage as the official versions:

```python
from langchain_core.messages import AIMessage
from langchain_dev_utils.agents.middleware import SummarizationMiddleware
from langchain_dev_utils.chat_models import register_model_provider

# Models must first be registered via register_model_provider
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
    system_prompt="You are an intelligent AI assistant capable of solving user problems",
)
response = agent.invoke({"messages": messages})
print(response)
```