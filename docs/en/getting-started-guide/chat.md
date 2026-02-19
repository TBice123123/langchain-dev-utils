# Chat Model Management

## Overview

LangChain's `init_chat_model` function only supports a limited number of model providers. This library provides a more flexible chat model management solution, supporting custom model providers, which is particularly useful for scenarios requiring integration with model services not natively supported (such as vLLM, etc.).

## Registering Model Providers

To register a chat model provider, call `register_model_provider`. The registration steps vary slightly depending on the situation.

### Existing LangChain Chat Model Class

If the model provider already has a ready-made and suitable LangChain integration (see [Chat Model Integrations](https://docs.langchain.com/oss/python/integrations/chat)), pass the corresponding integrated chat model class as the `chat_model` parameter.

#### Code Example

```python hl_lines="5 6"
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="fake_provider",
    chat_model=FakeChatModel,
)

# FakeChatModel is for testing only; in actual usage, you must pass a ChatModel class with real functionality.
```

!!! tip "Parameter Setting Instructions"
    `provider_name` represents the name of the model provider, used for subsequent reference in `load_chat_model`. The name must start with a letter or number, contain only letters, numbers, and underscores, and be no longer than 20 characters.

#### Optional Parameter Description

**base_url**

This parameter usually does not need to be set (since the chat model class generally defines a default API address internally). Only pass `base_url` when you need to override the default address of the chat model class, and it only takes effect for attributes named `api_base` or `base_url` (including aliases).

**model_profiles**

If your LangChain integrated chat model class fully supports the `profile` parameter (i.e., relevant model properties like `max_input_tokens`, `tool_calling`, etc., can be accessed directly via `model.profile`), there is no need to set `model_profiles` additionally.

If accessing via `model.profile` returns an empty dictionary `{}`, it indicates that the LangChain chat model class may not currently support the `profile` parameter. In this case, you can manually provide `model_profiles`.

`model_profiles` is a dictionary where each key is a model name, and the value is the profile configuration for that model:

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
    # Can contain any number of model configurations
}
```
!!! info "Hint"
    It is recommended to use the `langchain-model-profiles` library to obtain profiles for your model provider.

### No LangChain Chat Model Class, but Provider Supports OpenAI Compatible API

In this case, the `chat_model` parameter must be set to `"openai-compatible"`.

#### Code Example

```python hl_lines="2 3 4"
register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1"
)
```

**Note**: For more details on this part, please refer to [OpenAI Compatible API Integration](../adavance-guide/openai-compatible/register.md).


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
    Both registration functions are implemented based on a global dictionary. To avoid multi-threading issues, **all registrations must be completed during the application startup phase**; dynamic registration at runtime is prohibited.

    Additionally, if `chat_model` is set to `openai-compatible` during registration, a new model class is dynamically created internally via `pydantic.create_model` (generating the corresponding chat model integration class based on `BaseChatOpenAICompatible`). This process involves Python metaclass operations and pydantic validation logic initialization, which incurs some performance overhead. Therefore, please avoid frequent registration during runtime.


## Loading Chat Models

Use the `load_chat_model` function to load a chat model (initialize a chat model instance).

This function accepts a `model` parameter to specify the model name, an optional `model_provider` parameter to specify the model provider, and any number of keyword arguments to pass extra parameters to the chat model class.

#### Parameter Rules

- If `model_provider` is not passed, `model` must be in the `provider_name:model_name` format;
- If `model_provider` is passed, `model` must be just the `model_name`.

#### Code Example

```python
# Method 1: model includes provider info
model = load_chat_model("vllm:qwen2.5-7b")

# Method 2: Specify provider separately
model = load_chat_model("qwen2.5-7b", model_provider="vllm")
```

### Model Methods and Parameters

For supported model methods and parameters, refer to the usage instructions of the corresponding chat model class. If using the second scenario (OpenAI-compatible), all methods and parameters of the `BaseChatOpenAI` class are supported.

### Compatibility with Official Providers

`load_chat_model` looks up the global registration dictionary based on the `model_provider` parameter: if found, it instantiates the model class from that dictionary; if not found, it initializes via `init_chat_model`. This means providers officially supported by LangChain (like `openai`) can be called directly without registration.

```python
model = load_chat_model("openai:gpt-4o-mini")
# Or
model = load_chat_model("gpt-4o-mini", model_provider="openai")
```

!!! success "Best Practices"
    For the usage of this module, you can choose based on the following three situations:

    1. If all model providers being integrated are supported by the official `init_chat_model`, please use the official function directly for optimal compatibility and stability.

    2. If some model providers being integrated are not officially supported, you can use the features of this module: first register the model provider using `register_model_provider`, and then load the model using `load_chat_model`.

    3. If there is no suitable integration for the model provider yet, but the provider offers an OpenAI-compatible API (like vLLM), it is recommended to use the features of this module: first register the model provider using `register_model_provider` (pass `openai-compatible` to `chat_model`), and then load the model using `load_chat_model`.
