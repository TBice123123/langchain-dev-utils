# Integration with Model Management Functionality

This library seamlessly integrates this feature with model management functionality. When registering a chat model, simply set `chat_model` to `"openai-compatible"`; when registering an embedding model, set `embeddings_model` to `"openai-compatible"`.

## Chat Model Class Registration

The specific code is as follows:

**Method 1: Explicit parameter passing**

```python hl_lines="4 5 6"
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1"
)
```

**Method 2: Through environment variables (recommended for configuration management)**

```python hl_lines="4 5"
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible"
    # Automatically reads VLLM_API_BASE
)
```

Additionally, the `base_url`, `compatibility_options`, and `model_profiles` parameters from the `create_openai_compatible_model` function are also supported. You just need to pass the corresponding parameters in the `register_model_provider` function.

For example:

```python hl_lines="7-11"
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1",
    compatibility_options={
        "supported_tool_choice": ["auto", "none", "required", "specific"],
        "supported_response_format": ["json_schema"]
    },
    model_profiles=model_profiles,
)
```

## Embedding Model Class Registration

Similar to chat model class registration:

**Method 1: Explicit parameter passing**

```python hl_lines="4 5 6"
from langchain_dev_utils.embeddings import register_embeddings_provider

register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)
```

**Method 2: Environment variables (recommended)**

```bash
export VLLM_API_BASE=http://localhost:8000/v1
```

```python hl_lines="4 5"
from langchain_dev_utils.embeddings import register_embeddings_provider

register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible"
)
```
