# Chat Model Management

> [!NOTE]  
> **Feature Overview**: Provides a more efficient and convenient way to manage chat models, supporting multiple model providers.  
> **Prerequisites**: Understanding of LangChain [Chat Models](https://docs.langchain.com/oss/python/langchain/models).  
> **Estimated Reading Time**: 15 minutes.

## Overview

LangChain's `init_chat_model` function only supports a limited number of model providers. This library offers a more flexible chat model management solution that supports custom model providers, making it particularly suitable for scenarios where you need to integrate with model services that are not natively supported (such as vLLM, OpenRouter, etc.).

Using the chat model management feature of this library involves two steps:

1.  **Register a Model Provider**

Use `register_model_provider` to register a model provider. Its parameters are defined as follows:
<Params  
name="provider_name"  
type="string"  
description="The name of the model provider, used as an identifier for subsequent model loading."  
:required="true"  
:default="null"  
/>  
<Params  
name="chat_model"  
type="BaseChatModel | string"  
description="The chat model, which can be a ChatModel class or a string (currently supports 'openai-compatible')."  
:required="true"  
:default="null"  
/>  
<Params  
name="base_url"  
type="string"  
description="The API address of the model provider (optional. Valid for both types of `chat_model`, but primarily used when `chat_model` is the string 'openai-compatible')."  
:required="false"  
:default="null"  
/>  
<Params  
name="model_profiles"  
type="dict"  
description="Declares the supported features and related parameters for each model offered by the provider (optional, valid for both `chat_model` types). The corresponding configuration will be read based on `model_name` and written to `model.profile` (e.g., containing fields like `max_input_tokens`, `tool_calling`, etc.)."  
:required="false"  
:default="null"  
/>
<Params  
name="compatibility_options"  
type="dict"  
description="Compatibility options for the model provider (optional, valid only when `chat_model` is the string 'openai-compatible'). Used to declare the provider's support for OpenAI-compatible features (like `tool_choice` strategies, `response_format` types, etc.) to ensure correct functionality."  
:required="false"  
:default="null"  
/>

2.  **Load a Chat Model**

Use `load_chat_model` to instantiate a specific model. Its parameters are defined as follows:

<Params  
name="model"  
type="string"  
description="The name of the chat model."  
:required="true"  
:default="null"  
/>  
<Params  
name="model_provider"  
type="string"  
description="The name of the chat model provider."  
:required="false"  
:default="null"  
/>

**Note**: The `load_chat_model` function can also accept any number of keyword arguments, which can be used to pass additional parameters such as `temperature`, `max_tokens`, `extra_body`, etc.

## Registering a Model Provider

To register a chat model provider, call `register_model_provider`. The registration steps vary slightly depending on the situation.

<CaseItem :step="1" content="Existing LangChain Chat Model Class"></CaseItem>

If the model provider already has a suitable and existing LangChain integration (see [Chat Model Integrations](https://docs.langchain.com/oss/python/integrations/chat)), pass the corresponding integrated chat model class as the `chat_model` parameter.

Refer to the following code for an example:
```python
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="fake_provider",
    chat_model=FakeChatModel,
)
```
A few additional notes on the code above:

- **`FakeChatModel` is for testing purposes only**. In actual use, you must pass in a `ChatModel` class with real functionality.
- **`provider_name` represents the name of the model provider**. It is used for reference in `load_chat_model` later. The name can be customized but should not contain special characters like ":" or "-".

Besides this, the function also accepts the following parameters in this case:

- **base_url**

**This parameter usually does not need to be set (since the model class typically defines a default API address)**. Only pass `base_url` when you need to override the model class's default address, and it only takes effect for attributes named `api_base` or `base_url` (including aliases).

- **model_profiles**

If your LangChain integrated chat model class fully supports the `profile` parameter (i.e., you can access the model's properties like `max_input_tokens`, `tool_calling`, etc., directly via `model.profile`), there is no need to set `model_profiles` additionally.

If `model.profile` returns an empty dictionary `{}`, it means the LangChain chat model class may not support the `profile` parameter yet. In this case, you can manually provide `model_profiles`.

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
    # You can have any number of model configurations
}
```

**Tip**: It is recommended to use the `langchain-model-profiles` library to get profiles for your model provider.

<CaseItem :step="2" content="No LangChain Chat Model Class, but Provider has an OpenAI-Compatible API"></CaseItem>

Many model providers offer services with an **OpenAI-compatible API**, such as: [vLLM](https://github.com/vllm-project/vllm), [OpenRouter](https://openrouter.ai/), [Together AI](https://www.together.ai/), etc. When the model provider you are integrating does not have a suitable LangChain chat model class but offers an OpenAI-compatible API, you can consider this approach.

This library will use the built-in `BaseChatOpenAICompatible` class to construct a chat model class specific to the provider based on user input. This class inherits from `langchain-openai`'s `BaseChatOpenAI` and enhances it with the following capabilities:

- **Support for more reasoning content formats**: Compared to `ChatOpenAI`, which only outputs official reasoning content, this class also supports outputting reasoning content in more formats (e.g., from `vLLM`).
- **Dynamic adaptation and selection of the optimal structured output method**: By default, it can automatically select the best structured output method (`function_calling` or `json_schema`) based on the actual support of the model provider.
- **Fine-grained adaptation of differences via `compatibility_options`**: By configuring provider compatibility options, it resolves support differences for parameters like `tool_choice` and `response_format`.

**Note**: When using this approach, you must install the `standard` version of the `langchain-dev-utils` library. For details, refer to [Installation](../installation.md).

In this case, besides passing the `provider_name` and `chat_model` (which must be `"openai-compatible"`) parameters, you also need to pass the `base_url` parameter.

For the `base_url` parameter, you can provide it in one of the following ways:

  - **Explicitly as an argument**:
    ```python
    register_model_provider(
        provider_name="vllm",
        chat_model="openai-compatible",
        base_url="http://localhost:8000/v1"
    )
    ```
  - **Via environment variable** (recommended for configuration management):
    ```bash
    export VLLM_API_BASE=http://localhost:8000/v1
    ```
    The `base_url` can be omitted in the code:
    ```python
    register_model_provider(
        provider_name="vllm",
        chat_model="openai-compatible"
        # Automatically reads VLLM_API_BASE
    )
    ```
    ::: info Tip
    In this case, the naming convention for the environment variable of the provider's API endpoint is `${PROVIDER_NAME}_API_BASE` (all caps, underscore-separated). The corresponding API_KEY environment variable follows the convention `${PROVIDER_NAME}_API_KEY` (all caps, underscore-separated).
    :::

::: tip Additional Information

vLLM is a popular large model inference framework that can deploy large models with an OpenAI-compatible API. For example, here is Qwen3-4B:

```bash
vllm serve Qwen/Qwen3-4B \
--reasoning-parser qwen3 \
--enable-auto-tool-choice --tool-call-parser hermes \
--host 0.0.0.0 --port 8000 \
--served-model-name qwen3-4b
```

The service address is `http://localhost:8000/v1`.
:::

At the same time, in this case, you can also set the following two optional parameters:

- **model_profiles**

In this situation, if `model_profiles` is not manually set, `model.profile` will return an empty dictionary `{}`. Therefore, if you need to get configuration information for a specific model via `model.profile`, you must explicitly set `model_profiles`.

- **compatibility_options**

This is only valid in this case. It is used to **declare** the provider's support for certain **OpenAI API** features to improve compatibility and stability.
Currently, the following configuration options are supported:

- `supported_tool_choice`: List of supported `tool_choice` strategies, default is `["auto"]`.
- `supported_response_format`: List of supported `response_format` types (`json_schema`, `json_object`), default is `[]`.
- `reasoning_keep_policy`: Policy for keeping the `reasoning_content` field in historical messages passed to the model. Options are `never`, `current`, `all`. Default is `never`.
- `include_usage`: Whether to include `usage` information in the last streaming response chunk. Default is `True`.

::: info Note
Since different models from the same provider may have varying support for parameters like `tool_choice` and `response_format`, these four compatibility options will ultimately become **instance attributes** of the class. Values passed during registration serve as global defaults (representing the configuration supported by most of the provider's models). If fine-tuning is needed during loading, you can override them with parameters of the same name in `load_chat_model`.
:::

::: details supported_tool_choice

`tool_choice` is used to control whether and which external tools the large model calls during a response, to improve accuracy, reliability, and controllability. Common values include:

- `"auto"`: The model decides autonomously whether to call a tool (default behavior).
- `"none"`: Prohibits calling tools.
- `"required"`: Forces the call of at least one tool.
- A specific tool (in OpenAI-compatible APIs, specifically `{"type": "function", "function": {"name": "xxx"}}`).

Different providers support different ranges. To avoid errors, this library defaults `supported_tool_choice` to `["auto"]`. This means that when using `bind_tools`, the `tool_choice` parameter can only be passed as `auto`; other values will be filtered out.

If you need to support other `tool_choice` values, you must configure the supported items. The configuration value is a list of strings, with each optional value being:

- `"auto"`, `"none"`, `"required"`: Correspond to standard strategies.
- `"specific"`: A unique identifier for this library, indicating support for specifying a particular tool.

For example, vLLM supports all strategies:

```python
register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    compatibility_options={"supported_tool_choice": ["auto", "none", "required", "specific"]},
)
```

::: info Tip
If there are no special requirements, you can keep the default (i.e., `["auto"]`). If your business scenario requires the model to **call a specific tool** or **choose one from a given list**, and the provider supports the corresponding strategy, you can enable it as needed:
1. If you require **at least one tool** to be called and the provider supports `required`, you can set it to `["required"]` (and when calling `bind_tools`, you need to explicitly pass `tool_choice="required"`).
2. If you require a **specific tool** to be called and the provider supports specifying a particular tool, you can set it to `["specific"]` (This configuration is very useful in `function_calling` structured output, as it ensures the model calls the specified structured output tool, guaranteeing the stability of the output. Because in the `with_structured_output` method, its internal implementation will pass a `tool_choice` value that forces the call to a specific tool when calling `bind_tools`. However, if `"specific"` is not in `supported_tool_choice`, this parameter will be filtered out. Therefore, to ensure `tool_choice` can be passed correctly, you must add `"specific"` to `supported_tool_choice`.)

This parameter can be set once in `register_model_provider` for the provider or overridden dynamically for a single model in `load_chat_model`. It is recommended to declare the `tool_choice` support for most of the provider's models in `register_model_provider`, and for models with different support, specify it separately in `load_chat_model`.
:::

::: details supported_response_format

Currently, there are three common methods for structured output.

- `function_calling`: Generates structured output by calling a tool that conforms to a specified schema.
- `json_schema`: A feature provided by the model provider specifically for generating structured output. In OpenAI-compatible APIs, this is `response_format={"type": "json_schema", "json_schema": {...}}`.
- `json_mode`: A feature offered by some providers before `json_schema` was available. It can generate valid JSON, but the schema must be described in the prompt. In OpenAI-compatible APIs, this is `response_format={"type": "json_object"}`.

Among these, `json_schema` is supported by only a few OpenAI-compatible API providers (e.g., `OpenRouter`, `TogetherAI`); `json_mode` has wider support and is compatible with most providers; while `function_calling` is the most universal, usable as long as the model supports tool calling.

This parameter is used to declare the model provider's support for `response_format`. By default, it is `[]`, meaning the provider supports neither `json_mode` nor `json_schema`. In this case, the `method` parameter in the `with_structured_output` method can only be `function_calling` (or `auto`, which will be inferred as `function_calling`). If `json_mode` or `json_schema` is passed, it will be automatically converted to `function_calling`. If you want to enable the `json_mode` or `json_schema` implementation for structured output, you need to explicitly set this parameter.

For example, if most models on OpenRouter support both `json_mode` and `json_schema` for `response_format`, you can declare this during registration:

```python
register_model_provider(
    provider_name="openrouter",
    chat_model="openai-compatible",
    compatibility_options={"supported_response_format": ["json_mode", "json_schema"]},
)
```

::: info Tip
Usually, there is no need to configure this. It only needs to be considered when using the `with_structured_output` method. If the model provider supports `json_schema`, you can consider configuring this parameter to ensure the stability of structured output. For `json_mode`, since it only guarantees JSON output, it is generally not necessary to set it. Only set this parameter to include `json_mode` when the model does not support tool calling and only supports setting `response_format={"type":"json_object"}`.

Similarly, this parameter can be set globally in `register_model_provider` or overridden for a single model in `load_chat_model`. It is recommended to declare the `response_format` support for most of the provider's models in `register_model_provider`, and for models with different support, specify it separately in `load_chat_model`.
::: warning Note
This parameter currently only affects the `model.with_structured_output` method. For structured output in `create_agent`, if you need to use the `json_schema` implementation, you need to ensure the model's `profile` contains the `structured_output` field with a value of `True`.
:::

::: details reasoning_keep_policy

Used to control the retention policy for the `reasoning_content` field in historical messages.

The following values are supported:
- `never`: **Do not retain any** reasoning content in historical messages (default).
- `current`: Only retain the `reasoning_content` field from the **current conversation turn**.
- `all`: Retain the `reasoning_content` field from **all conversation turns**.

For example:
A user first asks "What's the weather in New York?", then follows up with "What's the weather in London?". The current turn is the second conversation, and the final model call is about to be made.

- When the value is `never`

With `never`, the final `messages` passed to the model will **not contain any** `reasoning_content` fields. The final messages received by the model are:

```python
messages = [
    {"content": "What's the weather in New York?", "role": "user"},
    {"content": "", "role": "assistant", "tool_calls": [...]},
    {"content": "Cloudy, 7~13°C", "role": "tool", "tool_call_id": "..."},
    {"content": "The weather in New York today is cloudy, 7~13°C.", "role": "assistant"},
    {"content": "What's the weather in London?", "role": "user"},
    {"content": "", "role": "assistant", "tool_calls": [...]},
    {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
]
```
- When the value is `current`

With `current`, only the `reasoning_content` field from the **current conversation turn** is retained. The final messages received by the model are:
```python
messages = [
    {"content": "What's the weather in New York?", "role": "user"},
    {"content": "", "role": "assistant", "tool_calls": [...]},
    {"content": "Cloudy, 7~13°C", "role": "tool", "tool_call_id": "..."},
    {"content": "The weather in New York today is cloudy, 7~13°C.", "role": "assistant"},
    {"content": "What's the weather in London?", "role": "user"},
    {
        "content": "",
        "reasoning_content": "To check London's weather, I need to call the weather tool directly.",  # Only reasoning from this turn is kept
        "role": "assistant",
        "tool_calls": [...],
    },
    {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
]
```
- When the value is `all`

With `all`, the `reasoning_content` field from **all conversation turns** is retained. The final messages received by the model are:
```python
messages = [
    {"content": "What's the weather in New York?", "role": "user"},
    {
        "content": "",
        "reasoning_content": "To check New York's weather, I need to call the weather tool directly.",  # Reasoning is kept
        "role": "assistant",
        "tool_calls": [...],
    },
    {"content": "Cloudy, 7~13°C", "role": "tool", "tool_call_id": "..."},
    {
        "content": "The weather in New York today is cloudy, 7~13°C.",
        "reasoning_content": "Return the New York weather result directly.",  # Reasoning is kept
        "role": "assistant",
    },
    {"content": "What's the weather in London?", "role": "user"},
    {
        "content": "",
        "reasoning_content": "To check London's weather, I need to call the weather tool directly.",  # Reasoning is kept
        "role": "assistant",
        "tool_calls": [...],
    },
    {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
]
```

**Note**: If the current conversation turn does not involve a tool call, the effect of `current` is the same as `never`.
::: info Tip
Configure this flexibly based on the provider's requirements for retaining `reasoning_content`:
- If the provider requires reasoning content to be kept **throughout**, set it to `all`.
- If it only needs to be kept for the **current tool call**, set it to `current`.
- If there are no special requirements, keep the default `never`.

Similarly, this parameter can be set globally in `register_model_provider` or overridden for a single model in `load_chat_model`. If only a few models require retaining `reasoning_content`, it is recommended to specify it separately in `load_chat_model`, in which case `register_model_provider` does not need to set it.
:::

::: details include_usage

`include_usage` is a parameter in OpenAI-compatible APIs that controls whether to append a final message containing token usage information (like `prompt_tokens` and `completion_tokens`) to a streaming response. Since standard streaming responses do not return usage information by default, enabling this option allows the client to directly obtain complete token consumption data, which is useful for billing, monitoring, or logging.

It is typically enabled via `stream_options={"include_usage": true}`. Considering that some model providers do not support this parameter, this library makes it a compatibility option with a default value of `True`, as most providers support it. If a provider does not, you can explicitly set it to `False`.

::: info Tip
This parameter generally does not need to be set; keep the default value. Only set it to `False` if the model provider does not support it.
:::

::: warning Note
Despite the compatibility configurations provided above, this library cannot guarantee 100% compatibility with all OpenAI-compatible interfaces. If a model provider has an official or community-supported integration class, please prioritize using that integration. If you encounter any compatibility issues, feel free to submit an issue in this library's GitHub repository.
:::

## Batch Registration

If you need to register multiple providers, you can use `batch_register_model_provider` to avoid repetitive calls.

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

::: warning Note  
Both registration functions are implemented based on a global dictionary. To avoid multi-threading issues, **all registrations must be completed during the application startup phase**. Dynamic registration at runtime is prohibited.  
:::

## Loading a Chat Model

Use the `load_chat_model` function to load a chat model (initialize a chat model instance). The parameter rules are as follows:

- If `model_provider` is not passed, `model` must be in the format `provider_name:model_name`.
- If `model_provider` is passed, `model` must be just `model_name`.

**Examples**:

```python
# Method 1
model = load_chat_model("vllm:qwen3-4b")

# Method 2
model = load_chat_model("qwen3-4b", model_provider="vllm")
```

Although vLLM does not strictly require an API Key, LangChain still requires one to be set. You can set it in an environment variable:

```bash
export VLLM_API_KEY=vllm
```

### Model Methods and Parameters

For **Case 1** (existing LangChain class), all its methods and parameters are consistent with the corresponding chat model class.  
For **Case 2** (`openai-compatible`), the model's methods and parameters are as follows:

- Supports `invoke`, `ainvoke`, `stream`, `astream` and other methods.
- Supports the `bind_tools` method for tool calling.
- Supports the `with_structured_output` method for structured output.
- Supports passing parameters from `BaseChatOpenAI`, such as `temperature`, `top_p`, `max_tokens`, etc.
- Supports passing multimodal data.
- Supports OpenAI's latest `responses API`.
- Supports the `model.profile` parameter to get the model's profile.

:::details Basic Invocation

Supports `invoke` for simple calls:

```python
from langchain_dev_utils.chat_models import load_chat_model
from langchain_core.messages import HumanMessage

model = load_chat_model("vllm:qwen3-4b")
response = model.invoke([HumanMessage("Hello")])
print(response)
```

It also supports `ainvoke` for asynchronous calls:

```python
from langchain_dev_utils.chat_models import load_chat_model
from langchain_core.messages import HumanMessage

model = load_chat_model("vllm:qwen3-4b")
response = await model.ainvoke([HumanMessage("Hello")])
print(response)
```

:::

:::details Streaming Output

Supports `stream` for streaming output:

```python
from langchain_dev_utils.chat_models import load_chat_model
from langchain_core.messages import HumanMessage

model = load_chat_model("vllm:qwen3-4b")
for chunk in model.stream([HumanMessage("Hello")]):
    print(chunk)
```

And `astream` for asynchronous streaming:

```python
from langchain_dev_utils.chat_models import load_chat_model
from langchain_core.messages import HumanMessage

model = load_chat_model("vllm:qwen3-4b")
async for chunk in model.astream([HumanMessage("Hello")]):
    print(chunk)
```

:::

:::details Tool Calling

If the model itself supports tool calling, you can use the `bind_tools` method directly:

```python
from langchain_dev_utils.chat_models import load_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import datetime

@tool
def get_current_time() -> str:
    """Get the current timestamp"""
    return str(datetime.datetime.now().timestamp())

model = load_chat_model("vllm:qwen3-4b").bind_tools([get_current_time])
response = model.invoke([HumanMessage("Get the current timestamp")])
print(response)
```

:::

:::details Structured Output

Supports structured output. The default `method` is `auto`, which will automatically select the appropriate structured output method based on the provider's `supported_response_format` parameter. Specifically, if `json_schema` is included in the values, the `json_schema` method will be chosen; otherwise, the `function_calling` method will be used.

```python
from langchain_dev_utils.chat_models import load_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

model = load_chat_model("vllm:qwen3-4b").with_structured_output(User)
response = model.invoke([HumanMessage("Hello, my name is Zhang San, I am 25 years old")])
print(response)
```
Compared to tool calling, `json_schema` can 100% guarantee that the output conforms to the JSON Schema, avoiding potential parameter errors from tool calling. Therefore, if the model provider supports `json_schema`, it will be used by default. When the provider does not support it, it will fall back to the `function_calling` method.
For `json_mode`, although it has wider support, it is more cumbersome to use because the schema must be described in the prompt to guide the model to output a JSON string. Therefore, it is not used by default. If you want to use it, you can explicitly provide `method="json_mode"` (provided that `supported_response_format` includes `json_mode` during registration or instantiation).
:::

:::details Passing Model Parameters

Additionally, since this class inherits from `BaseChatOpenAI`, it supports passing model parameters from `BaseChatOpenAI`, such as `temperature`, `extra_body`, etc.:

```python
from langchain_dev_utils.chat_models import load_chat_model
from langchain_core.messages import HumanMessage

model = load_chat_model("vllm:qwen3-4b", extra_body={"chat_template_kwargs": {"enable_thinking": False}}) # Use extra_body to pass additional parameters, here to disable thinking mode
response = model.invoke([HumanMessage("Hello")])
print(response)
```

:::

:::details Passing Multimodal Data

Supports passing multimodal data. You can use the OpenAI-compatible multimodal data format or directly use `content_block` from `langchain`. For example:

```python
from langchain_dev_utils.chat_models import register_model_provider, load_chat_model
from langchain_core.messages import HumanMessage


register_model_provider(
    provider_name="openrouter",
    chat_model="openai-compatible",
    base_url="https://openrouter.ai/api/v1",
)

messages = [
    HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
            {"type": "text", "text": "Describe this image"},
        ]
    )
]

model = load_chat_model("openrouter:qwen/qwen3-vl-8b-thinking")
response = model.invoke(messages)
print(response)
```

:::

:::details OpenAI's Latest `responses_api`

This model class also supports OpenAI's latest `responses_api`. However, currently only a few providers support this API style. If your model provider supports this API style, you can pass `use_responses_api=True`.
For example, vLLM supports `responses_api`, so you can use it like this:

```python
from langchain_dev_utils.chat_models import load_chat_model
from langchain_core.messages import HumanMessage

model = load_chat_model("vllm:qwen3-4b", use_responses_api=True)
response = model.invoke([HumanMessage(content="Hello")])
print(response)
```

:::

:::details Getting Model Profile
Taking OpenRouter as an example, you first need to install the `langchain-model-profiles` library:

```bash
pip install langchain-model-profiles
```

Then you can use the following command to get the model profiles supported by OpenRouter:

```bash
langchain-profiles refresh --provider openrouter --data-dir ./data/openrouter
```

This will generate a `_profiles.py` file in the `./data/openrouter` folder in your project's root directory. This file contains a dictionary variable named `_PROFILES`.

Next, you can refer to the following example for the code:

```python
from langchain_dev_utils.chat_models import load_chat_model, register_model_provider

from data.openrouter._profiles import _PROFILES

register_model_provider("openrouter", "openai-compatible", model_profiles=_PROFILES)

model = load_chat_model("openrouter:openai/gpt-5-mini")
print(model.profile)
```

:::

### Compatibility with Official Providers

For providers officially supported by LangChain (like `openai`), you can use `load_chat_model` directly without registration:

```python
model = load_chat_model("openai:gpt-4o-mini")
# or
model = load_chat_model("gpt-4o-mini", model_provider="openai")
```

<BestPractice>
    <p>For using this module, you can choose based on the following three scenarios:</p>
    <ol>
        <li>If all model providers you are integrating are supported by the official <code>init_chat_model</code>, please use the official function directly for the best compatibility and stability.</li>
        <li>If some of the model providers you are integrating are not officially supported, you can use this module's features. First, use <code>register_model_provider</code> to register the model providers, then use <code>load_chat_model</code> to load the models.</li>
        <li>If the model providers you are integrating do not have a suitable integration but offer an OpenAI-compatible API (like vLLM, OpenRouter), it is recommended to use this module's features. First, use <code>register_model_provider</code> to register the model providers (passing <code>openai-compatible</code> for `chat_model`), then use <code>load_chat_model</code> to load the models.</li>
    </ol>
</BestPractice>