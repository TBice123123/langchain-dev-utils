# Chat Model Management

## Overview

LangChain's `init_chat_model` function only supports a limited number of model providers. This library offers a more flexible chat model management solution that supports custom model providers, particularly suitable for scenarios where you need to integrate model services not natively supported (such as vLLM).

## Registering Model Providers

To register a chat model provider, call `register_model_provider`. The registration steps vary slightly for different situations.

### Existing LangChain Chat Model Class

If the model provider already has a suitable LangChain integration (see [Chat Model Class Integration](https://docs.langchain.com/oss/python/integrations/chat)), pass the corresponding integrated chat model class as the chat_model parameter.

#### Parameter Description

| Parameter | Description |
|-----------|-------------|
| `provider_name` | Model provider name, used for subsequent reference in `load_chat_model`.<br><br>**Type**: `str`<br>**Required**: Yes |
| `chat_model` | LangChain chat model class.<br><br>**Type**: `type[BaseChatModel]`<br>**Required**: Yes |
| `base_url` | API base URL, usually no need to set manually.<br><br>**Type**: `str`<br>**Required**: No |
| `model_profiles` | Dictionary of model configuration information.<br><br>**Type**: `dict`<br>**Required**: No |

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
- `provider_name` represents the name of the model provider, used for subsequent reference in `load_chat_model`. The name can be customized, but should not contain special characters such as `:`, `-`, etc.

#### Optional Parameter Description

**base_url**

This parameter usually doesn't need to be set (since the model class generally has a default API address defined internally). It should only be passed when you need to override the model class's default address, and it only affects attributes with field names `api_base` or `base_url` (including aliases).

**model_profiles**

If your LangChain integrated chat model class fully supports the `profile` parameter (i.e., you can directly access model-related properties through `model.profile`, such as `max_input_tokens`, `tool_calling`, etc.), there's no need to set `model_profiles` additionally.

If accessing through `model.profile` returns an empty dictionary `{}`, it indicates that the LangChain chat model class may not support the `profile` parameter yet, in which case you can manually provide `model_profiles`.

`model_profiles` is a dictionary where each key is a model name, and the value is the profile configuration for the corresponding model:

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
    # you can have any number of model configurations
}
```
!!! info "Tip"
    It's recommended to use the `langchain-model-profiles` library to get profiles for your model provider.

### No LangChain Chat Model Class, but Provider Supports OpenAI Compatible API

The parameter description for this situation is as follows:

#### Parameter Description

| Parameter | Description |
|-----------|-------------|
| `provider_name` | Model provider name.<br><br>**Type**: `str`<br>**Required**: Yes |
| `chat_model` | Fixed value `"openai-compatible"`.<br><br>**Type**: `str`<br>**Required**: Yes |
| `base_url` | API base URL.<br><br>**Type**: `str`<br>**Required**: No |
| `model_profiles` | Dictionary of model configuration information.<br><br>**Type**: `dict`<br>**Required**: No |
| `compatibility_options` | Compatibility options configuration.<br><br>**Type**: `dict`<br>**Required**: No |

#### Code Example

**Method 1: Explicit Parameters**

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

| Parameter | Description |
|-----------|-------------|
| `providers` | List of provider configurations, each dictionary contains registration parameters.<br><br>**Type**: `list[dict]`<br>**Required**: Yes |

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
    Both registration functions are implemented based on a global dictionary. To avoid multi-threading issues, **all registrations must be completed during the application startup phase**, and dynamic registration during runtime is prohibited.

    Additionally, when registering with `chat_model` set to `openai-compatible`, the system internally uses `pydantic.create_model` to dynamically create new model classes (with `BaseChatOpenAICompatible` as the base class, generating corresponding chat model integration classes). This process involves Python metaclass operations and pydantic validation logic initialization, which has certain performance overhead, so please avoid frequent registration during runtime.

## Loading Chat Models

Use the `load_chat_model` function to load chat models (initialize chat model instances).

#### Parameter Description

| Parameter | Description |
|-----------|-------------|
| `model` | Model name.<br><br>**Type**: `str`<br>**Required**: Yes |
| `model_provider` | Model provider name.<br><br>**Type**: `str`<br>**Required**: No |

**In addition, any number of keyword arguments can be passed to provide additional parameters for the chat model class.**

#### Parameter Rules

- If `model_provider` is not passed, then `model` must be in the format `provider_name:model_name`;
- If `model_provider` is passed, then `model` must only be `model_name`.

#### Code Example

```python
# Method 1: model includes provider information
model = load_chat_model("vllm:qwen3-4b")

# Method 2: specify provider separately
model = load_chat_model("qwen3-4b", model_provider="vllm")
```

### Model Methods and Parameters

For supported model methods and parameters, refer to the usage instructions of the corresponding chat model class. If you're using the second situation, all methods and parameters of the `BaseChatOpenAI` class are supported.

### Compatibility with Official Providers

For providers already supported by LangChain (such as `openai`), you can directly use `load_chat_model` without registration:

```python
model = load_chat_model("openai:gpt-4o-mini")
# or
model = load_chat_model("gpt-4o-mini", model_provider="openai")
```

!!! success "Best Practice"
    For the use of this module, you can choose based on the following three situations:

    1. If all model providers you're integrating are supported by the official `init_chat_model`, please use the official function directly for the best compatibility and stability.

    2. If some model providers you're integrating are not officially supported, you can use the functionality of this module. First, use `register_model_provider` to register the model provider, then use `load_chat_model` to load the model.

    3. If the model provider you're integrating doesn't have a suitable integration yet, but the provider offers an OpenAI-compatible API (such as vLLM), it's recommended to use the functionality of this module. First, use `register_model_provider` to register the model provider (passing `openai-compatible` for chat_model), then use `load_chat_model` to load the model.