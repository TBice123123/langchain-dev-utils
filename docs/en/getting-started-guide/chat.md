# Chat Model Management

## Overview

LangChain's `init_chat_model` function only supports a limited number of model providers. This library provides a more flexible chat model management solution that supports custom model providers, particularly suitable for scenarios requiring integration with model services not natively supported (such as vLLM, etc.).

## Registering Model Providers

Registering a chat model provider requires calling `register_model_provider`. The registration steps vary slightly depending on the situation.

### Existing LangChain Chat Model Classes

If the model provider already has a ready-made and suitable LangChain integration (see [Chat Model Class Integrations](https://docs.langchain.com/oss/python/integrations/chat)), pass the corresponding integrated chat model class as the `chat_model` parameter.

#### Code Example

```python hl_lines="5 6"
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="fake_provider",
    chat_model=FakeChatModel,
)

# FakeChatModel is for testing only. In actual usage, you must pass a ChatModel class with real functionality.
```

!!! tip "Parameter Setting Instructions"
    `provider_name` represents the name of the model provider, used for subsequent referencing in `load_chat_model`. The name must start with a letter or number, can only contain letters, numbers, and underscores, with a maximum length of 20 characters.

#### Optional Parameter Description

**base_url**

This parameter typically does not need to be set (because chat model classes usually define a default API address internally). Pass `base_url` only when you need to override the chat model class's default address, and it only affects attributes with field names `api_base` or `base_url` (including aliases).

**model_profiles**

If your LangChain integrated chat model class fully supports the `profile` parameter (i.e., you can directly access the model's related properties via `model.profile`, such as `max_input_tokens`, `tool_calling`, etc.), you do not need to set `model_profiles` additionally.

If accessing `model.profile` returns an empty dictionary `{}`, it indicates that the LangChain chat model class may not yet support the `profile` parameter. In this case, you can manually provide `model_profiles`.

`model_profiles` is a dictionary where each key is a model name and the value is the corresponding model's profile configuration:

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
    # Can have any number of model configurations
}
```
!!! info "Tip"
    It is recommended to use the `langchain-model-profiles` library to obtain profiles for the model providers you use.

### No Existing LangChain Chat Model Class, but Model Provider Supports OpenAI-Compatible API

In this case, the `chat_model` parameter must be set to `"openai-compatible"`.

#### Code Example

```python hl_lines="2 3 4"
register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1"
)
```

**Note**: For more details about this part, please refer to [OpenAI-Compatible API Integration](../adavance-guide/openai-compatible/register.md).

## Batch Registration

If you need to register multiple providers, you can use `batch_register_model_provider` to avoid repeated calls.

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
    Both registration functions are implemented based on a global dictionary. To avoid multi-threading issues, **all registrations must be completed during application startup**; dynamic registration during runtime is prohibited.

    Additionally, when setting `chat_model` to `openai-compatible` during registration, the library dynamically creates a new model class internally using `pydantic.create_model` (generating the corresponding chat model integration class based on `BaseChatOpenAICompatible`). This process involves Python metaclass operations and pydantic validation logic initialization, incurring some performance overhead. Therefore, avoid frequent registration during runtime.

## Loading Chat Models

Use the `load_chat_model` function to load a chat model (initialize a chat model instance).

This function receives the `model` parameter to specify the model name, the optional `model_provider` parameter to specify the model provider, and can also accept any number of keyword arguments for passing additional parameters to the chat model class.

#### Parameter Rules

- If `model_provider` is not passed, `model` must be in the format `provider_name:model_name`;
- If `model_provider` is passed, `model` must be only `model_name`.

#### Code Example

```python
# Method 1: model includes provider information
model = load_chat_model("vllm:qwen3-4b")

# Method 2: separately specify provider
model = load_chat_model("qwen3-4b", model_provider="vllm")
```

### Model Methods and Parameters

For supported model methods and parameters, refer to the usage documentation of the corresponding chat model class. If you are using the second case, all methods and parameters of the `BaseChatOpenAI` class are supported.

### Compatibility with Official Providers

`load_chat_model` looks up the global registration dictionary based on the `model_provider` parameter: if found, it instantiates using the corresponding model class from the dictionary; if not found, it initializes via `init_chat_model`. This means providers officially supported by LangChain (e.g., openai) can be called directly without registration.

```python
model = load_chat_model("openai:gpt-4o-mini")
# or
model = load_chat_model("gpt-4o-mini", model_provider="openai")
```

!!! success "Best Practice"
    For using this module, you can choose based on the following three situations:

    1. If all model providers you are integrating are supported by the official `init_chat_model`, use the official function directly for the best compatibility and stability.

    2. If some model providers you are integrating are not officially supported, use this module's functionality: first register the model provider using `register_model_provider`, then load the model using `load_chat_model`.

    3. If the model provider you are integrating does not have a suitable integration yet, but the provider offers an OpenAI-compatible API (such as vLLM), it is recommended to use this module's functionality: first register the model provider using `register_model_provider` (passing `openai-compatible` for `chat_model`), then load the model using `load_chat_model`.