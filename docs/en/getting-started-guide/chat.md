# Chat Model Management

## Overview

LangChain's `init_chat_model` function only supports a limited number of model providers. This library provides a more flexible chat model management solution that supports custom model providers, especially suitable for scenarios where you need to integrate model services not natively supported (such as vLLM).

## Registering Model Providers

To register a chat model provider, you need to call `register_model_provider`. The registration steps vary slightly for different situations.

### Existing LangChain Chat Model Class

If the model provider already has a ready and suitable LangChain integration (see [Chat Model Class Integration](https://docs.langchain.com/oss/python/integrations/chat)), please pass the corresponding integrated chat model class as the chat_model parameter.

#### Parameter Description

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `provider_name` | `str` | Yes | - | Model provider name, used for reference in `load_chat_model` |
| `chat_model` | `type[BaseChatModel]` | Yes | - | LangChain chat model class |
| `base_url` | `str` | No | `None` | API base address, usually no need to set manually |
| `model_profiles` | `dict` | No | `None` | Model configuration information dictionary |

#### Code Example

```python
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="fake_provider",
    chat_model=FakeChatModel,
)
```

#### Usage Instructions

- `FakeChatModel` is only for testing. In actual use, you must pass a `ChatModel` class with real functionality.
- `provider_name` represents the name of the model provider, used for reference in `load_chat_model`. The name can be customized, but should not contain special characters like `:`, `-`, etc.

#### Optional Parameter Description

**base_url**

This parameter usually does not need to be set (because the model class generally already defines a default API address). Only pass `base_url` when you need to override the model class's default address, and it only works for attributes with field names `api_base` or `base_url` (including aliases).

**model_profiles**

If your LangChain integrated chat model class already fully supports the `profile` parameter (i.e., you can directly access model-related properties through `model.profile`, such as `max_input_tokens`, `tool_calling`, etc.), then there is no need to set `model_profiles` additionally.

If accessing through `model.profile` returns an empty dictionary `{}`, it indicates that this LangChain chat model class may not yet support the `profile` parameter, in which case you can manually provide `model_profiles`.

`model_profiles` is a dictionary where each key is a model name, and the value is the profile configuration of the corresponding model:

```python
{
    "model_name_1": {
        "max_input_tokens": 100_000,
        "tool_calling": True,
        "structured_output": True,
        # ... other optional fields
    },
    "model_name_2": {
        "max_input_tokens": 32768,
        "image_inputs": True,
        "tool_calling": False,
        # ... other optional fields
    },
    #可以有任意多个模型配置
}
```
!!! info "Tip"
    It is recommended to use the `langchain-model-profiles` library to get profiles for your model provider.

### No LangChain Chat Model Class, but Provider Supports OpenAI Compatible API

The parameter description for this situation is as follows:

#### Parameter Description

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `provider_name` | `str` | Yes | - | Model provider name |
| `chat_model` | `str` | Yes | - | Fixed value `"openai-compatible"` |
| `base_url` | `str` | No | `None` | API base address |
| `model_profiles` | `dict` | No | `None` | Model configuration information dictionary |
| `compatibility_options` | `dict` | No | `None` | Compatibility options configuration |

#### Code Example

**Method 1: Explicit Parameter Passing**

```python
register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1"
)
```

**Method 2: Through Environment Variables (Recommended for Configuration Management)**

```bash
export VLLM_API_BASE=http://localhost:8000/v1
```

```python
register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible"
    # Automatically reads VLLM_API_BASE
)
``` 

**Note**: For more details on this part, please refer to [OpenAI Compatible API Integration](../adavance-guide/openai-compatible.md).

## Batch Registration

If you need to register multiple providers, you can use `batch_register_model_provider` to avoid repeated calls.

#### Parameter Description

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `providers` | `list[dict]` | Yes | - | List of provider configurations, each dictionary contains registration parameters |

#### Code Example

```python
from langchain_dev_utils.chat_models import batch_register_model_provider
from langchain_core.language_models.fake_chat_models import FakeChatModel

batch_register_model_provider(
    providers=[
        {
            "provider_name": "fake_provider",
            "chat_model": FakeChatModel,
        },
        {
            "provider_name": "vllm",
            "chat_model": "openai-compatible",
            "base_url": "http://localhost:8000/v1",
        },
    ]
)
```

!!! warning "Note"
    Both registration functions are implemented based on global dictionaries. To avoid multithreading issues, **all registrations must be completed during the application startup phase**, and dynamic registration during runtime is prohibited.

    Additionally, when registering with `chat_model` set to `openai-compatible`, a new model class will be dynamically created internally through `pydantic.create_model` (with `BaseChatOpenAICompatible` as the base class, generating the corresponding chat model integration class). This process involves Python metaclass operations and pydantic validation logic initialization, which has some performance overhead, so please avoid frequent registration during runtime.

## Loading Chat Models

Use the `load_chat_model` function to load chat models (initialize chat model instances).

#### Parameter Description

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `str` | Yes | - | Model name |
| `model_provider` | `str` | No | `None` | Model provider name |

**In addition, you can pass any number of keyword arguments for additional parameters of the chat model class.**

#### Parameter Rules

- If `model_provider` is not passed, then `model` must be in the format `provider_name:model_name`;
- If `model_provider` is passed, then `model` must be only `model_name`.

#### Code Example

```python
# Method 1: model includes provider information
model = load_chat_model("vllm:qwen3-4b")

# Method 2: specify provider separately
model = load_chat_model("qwen3-4b", model_provider="vllm")
```

### Model Methods and Parameters

For supported model methods and parameters, please refer to the usage instructions of the corresponding chat model class. If you are using the second situation, then all methods and parameters of the `BaseChatOpenAI` class are supported.

### Compatibility with Official Providers

For providers already officially supported by LangChain (such as `openai`), you can directly use `load_chat_model` without registration:

```python
model = load_chat_model("openai:gpt-4o-mini")
# or
model = load_chat_model("gpt-4o-mini", model_provider="openai")
```

!!! success "Best Practice"
    For the use of this module, you can choose according to the following three situations:

    1. If all model providers you integrate are already supported by the official `init_chat_model`, please use the official function directly to get the best compatibility and stability.

    2. If some model providers you integrate are not officially supported, you can use the functionality of this module, first register model providers using `register_model_provider`, then use `load_chat_model` to load models.

    3. If the model provider you integrate does not have a suitable integration yet, but the provider provides an OpenAI compatible API (such as vLLM), it is recommended to use the functionality of this module, first register the model provider using `register_model_provider` (pass `openai-compatible` to chat_model), then use `load_chat_model` to load the model.