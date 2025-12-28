# Chat Model Management

## Overview

LangChain's `init_chat_model` function only supports a limited number of model providers. This library provides a more flexible chat model management solution that supports custom model providers, especially suitable for scenarios where you need to integrate with model services not natively supported (such as vLLM).

## Registering Model Providers

To register a chat model provider, you need to call `register_model_provider`. The registration steps vary slightly for different situations.

### Existing LangChain Chat Model Class

If the model provider already has a suitable LangChain integration (see [Chat Model Class Integration](https://docs.langchain.com/oss/python/integrations/chat)), pass the corresponding integrated chat model class as the chat_model parameter.

For specific code, refer to the following example:
```python
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="fake_provider",
    chat_model=FakeChatModel,
)
```
A few notes about the code above:

- `FakeChatModel` is only for testing. In actual use, you must pass a `ChatModel` class with real functionality.
- `provider_name` represents the name of the model provider, used for reference in `load_chat_model` later. The name can be customized but should not contain special characters such as ":" or "-".

In addition, in this case, the function also accepts the following parameters:

- **base_url**

**This parameter usually doesn't need to be set (because the model class generally has a default API address defined internally)**, only pass `base_url` when you need to override the model class's default address, and it only works for attributes with field names `api_base` or `base_url` (including aliases).

- **model_profiles**

If your LangChain integrated chat model class fully supports the `profile` parameter (i.e., you can access model-related properties directly through `model.profile`, such as `max_input_tokens`, `tool_calling`, etc.), then there's no need to set `model_profiles` additionally.

If accessing through `model.profile` returns an empty dictionary `{}`, it means the LangChain chat model class may not support the `profile` parameter yet, and you can manually provide `model_profiles`.

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
    # can have any number of model configurations
}
```
!!! info "Tip"
    It's recommended to use the `langchain-model-profiles` library to get profiles for your model provider.

### No LangChain Chat Model Class, but Provider Supports OpenAI Compatible API

Many model providers support **OpenAI Compatible API** services, such as: [vLLM](https://github.com/vllm-project/vllm), [OpenRouter](https://openrouter.ai/), [Together AI](https://www.together.ai/), etc. When the model provider you're integrating doesn't have a suitable LangChain chat model class but the provider supports OpenAI Compatible API, you can consider this scenario.

!!!tip "Tip"
    A common way to integrate with OpenAI Compatible API is to directly use `ChatOpenAI` from `langchain-openai`, just by passing `base_url` and `api_key`. However, this approach is only suitable for simple scenarios and has many compatibility issues: it cannot display reasoning chains (`reasoning_content`) of non-OpenAI official inference models, doesn't support video-type content_block, and has low default strategy coverage for structured output. Therefore, this library specifically provides this functionality to solve the above problems. Thus, for simpler scenarios (especially those with low compatibility requirements), you can completely use `ChatOpenAI` without using this feature.
    
This library will build a chat model class corresponding to a specific provider using the built-in `BaseChatOpenAICompatible` class based on user input. This class inherits from `BaseChatOpenAI` in `langchain-openai` and enhances the following capabilities:

- **Support for more formats of reasoning content**: Compared to `ChatOpenAI` which can only output official reasoning content, this class also supports outputting more formats of reasoning content (e.g., `vLLM`).
- **Support for `video` type content_block**: `ChatOpenAI` cannot convert `type=video` `content_block`, which this implementation now supports.
- **Dynamic adaptation and selection of the most suitable structured output method**: By default, it can automatically select the optimal structured output method (`function_calling` or `json_schema`) based on the actual support of the model provider.
- **Fine-tune compatibility differences through compatibility_options**: By configuring provider compatibility options, resolve support differences for parameters like `tool_choice`, `response_format`, etc.

**Note**: When using this scenario, you must install the standard version of the `langchain-dev-utils` library. For details, refer to the installation section.

In this case, besides passing `provider_name` and `chat_model` (must be `"openai-compatible"`) parameters, you also need to pass the `base_url` parameter.

For the `base_url` parameter, you can provide it in either of the following ways:

  - **Explicit parameter passing**:
    ```python
    register_model_provider(
        provider_name="vllm",
        chat_model="openai-compatible",
        base_url="http://localhost:8000/v1"
    )
    ```
  - **Through environment variables** (recommended for configuration management):
    ```bash
    export VLLM_API_BASE=http://localhost:8000/v1
    ```
    You can omit `base_url` in the code:
    ```python
    register_model_provider(
        provider_name="vllm",
        chat_model="openai-compatible"
        # Automatically reads VLLM_API_BASE
    )
    ``` 
!!! info "Tip"
    In this case, the naming rule for the environment variable of the model provider's API endpoint is `${PROVIDER_NAME}_API_BASE` (all uppercase, separated by underscores). The corresponding API_KEY environment variable naming rule is `${PROVIDER_NAME}_API_KEY` (all uppercase, separated by underscores).

!!! note "Additional Information"
    vLLM is a commonly used large model inference framework that can deploy large models as OpenAI Compatible APIs, such as Qwen3-4B in this example:

    ```bash
    vllm serve Qwen/Qwen3-4B \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3-4b
    ```
    The service address is `http://localhost:8000/v1`.  


At the same time, in this case, you can also set the following two optional parameters:

- **model_profiles**

In this case, if `model_profiles` is not manually set, `model.profile` will return an empty dictionary `{}`. Therefore, if you need to get configuration information of a specific model through `model.profile`, you must explicitly set `model_profiles` first.


- **compatibility_options**

Only effective in this case. Used to **declare** the provider's support for some features of the **OpenAI API** to improve compatibility and stability.
Currently supports the following configuration options:

- `supported_tool_choice`: List of supported `tool_choice` strategies, default is `["auto"]`;
- `supported_response_format`: List of supported `response_format` formats (`json_schema`, `json_object`), default is `[]`;
- `reasoning_keep_policy`: Retention policy for the `reasoning_content` field in historical messages (messages) passed to the model. Optional values are `never`, `current`, `all`. Default is `never`.
- `include_usage`: Whether to include `usage` information in the last streaming response result, default is `True`.

!!! info "Additional Information"
    Because different models from the same model provider have different support for parameters like `tool_choice`, `response_format`, etc. Therefore, these four compatibility options will ultimately become **instance attributes** of the class. Values can be passed during registration as global defaults (representing the configuration supported by most models of this provider), and can be overridden in `load_chat_model` when loading for fine-tuning.


??? note "1. supported_tool_choice"
    `tool_choice` is used to control whether and which external tools the large model calls when responding, to improve accuracy, reliability, and controllability. Common values include:

    - `"auto"`: The model decides autonomously whether to call tools (default behavior);
    - `"none"`: Prohibit tool calling;
    - `"required"`: Force calling at least one tool;
    - Specify a specific tool (in OpenAI Compatible API, specifically `{"type": "function", "function": {"name": "xxx"}}`).

    Different providers support different ranges. To avoid errors, this library defaults `supported_tool_choice` to `["auto"]`, which means when using `bind_tools`, the `tool_choice` parameter can only pass `auto`, and other values will be filtered out.

    If you need to support passing other `tool_choice` values, you must configure the supported items. The configuration value is a list of strings, with optional values for each string:

    - `"auto"`, `"none"`, `"required"`: Corresponding to standard strategies;
    - `"specific"`: A unique identifier in this library, indicating support for specifying a specific tool.

    For example, vLLM supports all strategies:

    ```python
    register_model_provider(
        provider_name="vllm",
        chat_model="openai-compatible",
        compatibility_options={"supported_tool_choice": ["auto", "none", "required", "specific"]},
    )
    ```

    !!! info "Tip"
        If there are no special requirements, you can keep the default (i.e., `["auto"]`). If your business scenario requires the model to **must call a specific tool** or **select one from a given list**, and the model provider supports the corresponding strategy, then enable as needed:
        
        1. If you require **at least one tool** to be called, and the model provider supports `required`, you can set it to `["required"]` (and at the same time, when calling `bind_tools`, you need to explicitly pass `tool_choice="required"`)

        2. If you require **calling a specific** tool, and the model provider supports specifying a specific tool call, you can set it to `["specific"]` (in `function_calling` structured output, this configuration is very useful to ensure the model calls the specified structured output tool to ensure the stability of structured output. Because in the `with_structured_output` method, its internal implementation will pass a `tool_choice` value that **forces calling the specified tool** when calling `bind_tools`, but if `"specific"` is not in `supported_tool_choice`, this parameter will be filtered out. Therefore, if you want to ensure that `tool_choice` can be passed normally, you must add `"specific"` to `supported_tool_choice`.)

        This parameter can be set uniformly in `register_model_provider` or dynamically overridden for a single model in `load_chat_model`; it's recommended to declare the `tool_choice` support for most models of this provider at once in `register_model_provider`, and for models with different support, specify them separately in `load_chat_model`.

??? note "2. supported_response_format"
    Currently, there are three common methods for structured output.

    - `function_calling`: Generate structured output by calling a tool that conforms to a specified schema.
    - `json_schema`: A feature provided by the model provider specifically for generating structured output. In OpenAI Compatible API, specifically `response_format={"type": "json_schema", "json_schema": {...}}`.
    - `json_mode`: A feature provided by some providers before they introduced `json_schema`, which can generate valid JSON, but the schema must be described in the prompt. In OpenAI Compatible API, specifically `response_format={"type": "json_object"}`).

    Among these, `json_schema` is only supported by a few OpenAI Compatible API providers (such as `OpenRouter`, `TogetherAI`); `json_mode` has higher support and is compatible with most providers; while `function_calling` is the most universal, as long as the model supports tool calling, it can be used.

    This parameter is used to declare the model provider's support for `response_format`. By default, it's `[]`, representing that the model provider neither supports `json_mode` nor `json_schema`. In this case, the `method` parameter in the `with_structured_output` method can only be `function_calling`. If `json_mode` or `json_schema` is passed, it will be automatically converted to `function_calling`. If you want to enable `json_mode` or `json_schema` structured output implementation, you need to explicitly set this parameter.

    For example, if the model deployed by vLLM supports the `json_schema` structured output method, you can declare it during registration:

    ```python
    register_model_provider(
        provider_name="vllm",
        chat_model="openai-compatible",
        compatibility_options={"supported_response_format": ["json_schema"]},
    )
    ``` 

    !!! info "Tip"
        Usually, there's no need to configure this. It's only necessary to consider configuring this parameter when using the `with_structured_output` method. If the model provider supports `json_schema`, you can consider configuring this parameter (because the stability of `json_schema` structured output is better than `function_calling`). For `json_mode`, since it only guarantees JSON output, it's generally not necessary to set it. Only when the model doesn't support tool calling and only supports setting `response_format={"type":"json_object"}`, do you need to configure this parameter to include `json_mode`.
        
        Similarly, this parameter can be set uniformly in `register_model_provider` or dynamically overridden for a single model in `load_chat_model`; it's recommended to declare the `response_format` support for most models of this provider at once in `register_model_provider`, and for models with different support, specify them separately in `load_chat_model`.

    !!! warning "Note"
        This parameter currently only affects the `model.with_structured_output` method. For structured output in `create_agent`, if you need to use the `json_schema` implementation, you need to ensure that the `profile` of the corresponding model includes the `structured_output` field, and its value is `True`.

??? note "3. reasoning_keep_policy"

    Used to control the retention policy for the `reasoning_content` field in historical messages (messages).

    Supports the following values:

    - `never`: **Do not retain any** reasoning content in historical messages (default);

    - `current`: Only retain the `reasoning_content` field in the **current conversation**;

    - `all`: Retain the `reasoning_content` field in **all conversations**.

    For example:
    For example, the user first asks "What's the weather in New York?", then follows up with "What's the weather in London?", currently in the second round of conversation, and about to make the last model call.

    - When the value is `never`

    When the value is `never`, then the final messages passed to the model will **not have any** `reasoning_content` fields. The final messages received by the model will be:

    ```python
    messages = [
        {"content": "What's the weather in New York?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Cloudy, 7~13°C", "role": "tool", "tool_call_id": "..."},
        {"content": "New York's weather today is cloudy, 7~13°C.", "role": "assistant"},
        {"content": "What's the weather in London?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    - When the value is `current`

    When the value is `current`, only the `reasoning_content` field in the **current conversation** is retained. The final messages received by the model will be:
    ```python
    messages = [
        {"content": "What's the weather in New York?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Cloudy, 7~13°C", "role": "tool", "tool_call_id": "..."},
        {"content": "New York's weather today is cloudy, 7~13°C.", "role": "assistant"},
        {"content": "What's the weather in London?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "To check London's weather, need to directly call the weather tool.",  # Only retain reasoning_content for this round of conversation
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    - When the value is `all`

    When the value is `all`, the `reasoning_content` field in **all** conversations is retained. The final messages received by the model will be:
    ```python
    messages = [
        {"content": "What's the weather in New York?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "To check New York's weather, need to directly call the weather tool.",  # Retain reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Cloudy, 7~13°C", "role": "tool", "tool_call_id": "..."},
        {
            "content": "New York's weather today is cloudy, 7~13°C.",
            "reasoning_content": "Directly return New York's weather result.",  # Retain reasoning_content
            "role": "assistant",
        },
        {"content": "What's the weather in London?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "To check London's weather, need to directly call the weather tool.",  # Retain reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    **Note**: If the current round of conversation doesn't involve tool calling, the effect of `current` is the same as `never`.

    !!! info "Tip"
        Configure flexibly according to the model provider's requirements for retaining `reasoning_content`:

        - If the provider requires **retaining reasoning content throughout**, set to `all`;  
        - If only required to retain in the **current tool call**, set to `current`;  
        - If there are no special requirements, keep the default `never`.

        Similarly, this parameter can be set uniformly in `register_model_provider` or dynamically overridden for a single model in `load_chat_model`; if there are few models that need to retain `reasoning_content`, it's recommended to specify them separately in `load_chat_model`, without setting in `register_model_provider`.



??? note "4. include_usage"

    `include_usage` is a parameter in OpenAI Compatible API used to control whether to append a message containing token usage information (such as `prompt_tokens` and `completion_tokens`) at the end of the streaming response. Since standard streaming responses don't return usage information by default, enabling this option allows clients to directly obtain complete token consumption data, facilitating billing, monitoring, or logging.

    Usually enabled through `stream_options={"include_usage": true}`. Considering that some model providers don't support this parameter, this library sets it as a compatibility option with a default value of `True`, because most model providers support this parameter. If not supported, it can be explicitly set to `False`.

    !!! info "Tip"
        This parameter generally doesn't need to be set, keep the default value. Only when the model provider doesn't support it, do you need to set it to `False`.



!!! warning "Note"
    Despite providing the above compatibility configurations, this library still cannot guarantee 100% compatibility with all OpenAI Compatible interfaces. If the model provider already has an official or community integration class, please prioritize using that integration class. If you encounter any compatibility issues, you're welcome to submit an issue in this library's GitHub repository.


## Batch Registration

If you need to register multiple providers, you can use `batch_register_model_provider` to avoid repeated calls.

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


## Loading Chat Models

Use the `load_chat_model` function to load chat models (initialize chat model instances). The parameter rules are as follows:

- If `model_provider` is not passed, then `model` must be in the format `provider_name:model_name`;
- If `model_provider` is passed, then `model` must only be `model_name`.

**Examples**:

```python
# Method 1
model = load_chat_model("vllm:qwen3-4b")

# Method 2
model = load_chat_model("qwen3-4b", model_provider="vllm")
```

Although vLLM doesn't strictly require an API Key, LangChain still requires it to be set. You can set it in environment variables:

```bash
export VLLM_API_KEY=vllm
```

### Model Methods and Parameters

For **case one**, all its methods and parameters are consistent with the corresponding chat model class.  
For **case two**, the model's methods and parameters are as follows:

- Supports `invoke`, `ainvoke`, `stream`, `astream` and other methods.


??? example "Regular Call"

    Supports `invoke` for simple calls:

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage

    model = load_chat_model("vllm:qwen3-4b")
    response = model.invoke([HumanMessage("Hello")])
    print(response)
    ```

    Also supports `ainvoke` for asynchronous calls:

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage

    model = load_chat_model("vllm:qwen3-4b")
    response = await model.ainvoke([HumanMessage("Hello")])
    print(response)
    ```


??? example "Streaming Output"

    Supports `stream` for streaming output:

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage

    model = load_chat_model("vllm:qwen3-4b")
    for chunk in model.stream([HumanMessage("Hello")]):
        print(chunk)
    ```

    And `astream` for asynchronous streaming calls:

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage

    model = load_chat_model("vllm:qwen3-4b")
    async for chunk in model.astream([HumanMessage("Hello")]):
        print(chunk)
    ```

- Supports the `bind_tools` method for tool calling.

If the model itself supports tool calling, you can directly use the `bind_tools` method for tool calling:

??? example "Tool Calling"

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    import datetime

    @tool
    def get_current_time() -> str:
        """Get current timestamp"""
        return str(datetime.datetime.now().timestamp())

    model = load_chat_model("vllm:qwen3-4b").bind_tools([get_current_time])
    response = model.invoke([HumanMessage("Get current timestamp")])
    print(response)
    ```

- Supports the `with_structured_output` method for structured output.

If the model class's `supported_response_format` parameter includes `json_schema`, then `with_structured_output` will prioritize using `json_schema` for structured output, otherwise fall back to `function_calling`; if you need `json_mode`, explicitly specify `method="json_mode"` and ensure registration includes `json_mode`.

??? example "Structured Output"

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int

    model = load_chat_model("vllm:qwen3-4b").with_structured_output(User)
    response = model.invoke([HumanMessage("Hello, my name is Zhang San, I'm 25 years old")])
    print(response)
    ```
    


- Supports passing parameters of `BaseChatOpenAI`, such as `temperature`, `top_p`, `max_tokens`, etc.

In addition, since this class inherits from `BaseChatOpenAI`, it supports passing model parameters of `BaseChatOpenAI`, such as `temperature`, `extra_body`, etc.:

??? example "Passing Model Parameters"

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage

    model = load_chat_model("vllm:qwen3-4b", extra_body={"chat_template_kwargs": {"enable_thinking": False}}) # Use extra_body to pass additional parameters, here to disable thinking mode
    response = model.invoke([HumanMessage("Hello")])
    print(response)
    ```


- Supports passing multimodal data

Supports passing multimodal data, you can use OpenAI Compatible multimodal data format or directly use `content_block` in `langchain`.

??? example "Passing Multimodal Data"

    **Passing image data**:

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage
    messages = [
        HumanMessage(
            content_blocks=[
                {
                    "type": "image",
                    "url": "https://example.com/image.png",
                },
                {"type": "text", "text": "Describe this image"},
            ]
        )
    ]

    model = load_chat_model("vllm:qwen3-vl-2b")
    response = model.invoke(messages)
    print(response)
    ```

    **Passing video data**:
    

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage

    messages = [
        HumanMessage(
            content_blocks=[
                {
                    "type": "video",
                    "url": "https://example.com/video.mp4",
                },
                {"type": "text", "text": "Describe this video"},
            ]
        )
    ]

    model = load_chat_model("vllm:qwen3-vl-2b")
    response = model.invoke(messages)
    print(response)
    ```
    
!!! note "Additional Information"
    vllm also supports deploying multimodal models, such as `qwen3-vl-2b`:
    ```bash
    vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3-vl-2b
    ```


- Supports OpenAI's latest `responses api` (not yet fully guaranteed, can be used for simple testing, but not for production environments)

This model class also supports OpenAI's latest `responses_api`. However, currently only a few providers support this API style. If your model provider supports this API style, you can pass `use_responses_api=True` as a parameter.
    For example, if vllm supports `responses_api`, you can use it like this:

??? example "OpenAI's latest `responses_api`"

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage

    model = load_chat_model("vllm:qwen3-4b", use_responses_api=True)
    response = model.invoke([HumanMessage(content="Hello")])
    print(response)
    ```

!!! info "Tip"
    In either case, you can pass any number of keyword arguments as additional parameters for the model, such as `temperature`, `extra_body`, etc.

### Compatibility with Official Providers

For providers already officially supported by LangChain (such as `openai`), you can directly use `load_chat_model` without registration:

```python
model = load_chat_model("openai:gpt-4o-mini")
# or
model = load_chat_model("gpt-4o-mini", model_provider="openai")
```

!!! success "Best Practice"
    For the use of this module, you can choose based on the following three situations:

    1. If all model providers you're integrating are supported by the official `init_chat_model`, please use the official function directly for the best compatibility and stability.

    2. If some of the model providers you're integrating are not officially supported, you can use the functionality of this module, first register the model provider using `register_model_provider`, then use `load_chat_model` to load the model.

    3. If the model provider you're integrating doesn't have a suitable integration but provides an OpenAI Compatible API (such as vLLM), it's recommended to use the functionality of this module, first register the model provider using `register_model_provider` (pass `openai-compatible` as chat_model), then use `load_chat_model` to load the model.