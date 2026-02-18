# 与模型管理功能集成

本库已将此功能无缝接入模型管理功能。注册对话模型时，只需将 `chat_model` 设为 `"openai-compatible"`；注册嵌入模型时，将 `embeddings_model` 设为 `"openai-compatible"` 即可。

## 对话模型类注册

具体代码如下：

**方式一：显式传参**

```python hl_lines="4 5 6"
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1"
)
```

**方式二：通过环境变量（推荐用于配置管理）**

```python hl_lines="4 5"
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible"
    # 自动读取 VLLM_API_BASE
)
```

同时，`create_openai_compatible_model`函数中的`base_url`、`compatibility_options`、`model_profiles`参数也支持传入。只需要在`register_model_provider`函数中传入对应的参数即可。

例如：

```python hl_lines="7-11"
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1",
    compatibility_options={
        "supported_tool_choice": ["auto", "none", "required", "specific"],
        "supported_response_format": ["json_schema"],
        "reasoning_field_name": "reasoning",
    },
    model_profiles=model_profiles,
)
```

## 嵌入模型类注册

与对话模型类注册类似：

**方式一：显式传参**

```python hl_lines="4 5 6"
from langchain_dev_utils.embeddings import register_embeddings_provider

register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)
```

**方式二：环境变量（推荐）**

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