# OpenAI Compatible API Model Provider Integration

## Overview

Many model providers support **OpenAI compatible** API services, such as: [vLLM](https://github.com/vllm-project/vllm), [OpenRouter](https://openrouter.ai/), [Together AI](https://www.together.ai/), etc. This library provides a complete OpenAI compatible API integration solution, supporting both chat models and embedding models, especially suitable for scenarios where there is no corresponding LangChain integration yet but the provider offers an OpenAI compatible API.

!!! tip "Tip"
    The common approach to integrate with OpenAI compatible APIs is to directly use `ChatOpenAI` or `OpenAIEmbeddings` from `langchain-openai`, simply by passing in `base_url` and `api_key`. However, this approach only works for simple scenarios and has many compatibility issues, especially for chat models, including:

    1. Unable to display reasoning content (`reasoning_content`) from non-OpenAI official inference models
    2. No support for video type content_block
    3. Low coverage rate for default structured output strategies

    This library provides this functionality to solve the above compatibility issues. For simple scenarios (especially those with low compatibility requirements), you can directly use `ChatOpenAI` without using this feature. `OpenAIEmbeddings` has good compatibility, just set `check_embedding_ctx_length` to `False`. Additionally, for developers' convenience, we also provide embedding model OpenAI compatible integration class functionality.

## Creating Corresponding Integration Classes

This library provides two utility functions for creating corresponding chat model integration classes and embedding model integration classes. Specifically:

| Function Name | Description |
|--------|------|
| `create_openai_compatible_model` | Create chat model integration class |
| `create_openai_compatible_embedding` | Create embedding model integration class |

### Creating Chat Model Class

You can use the `create_openai_compatible_model` function to create a chat model integration class. This function accepts the following parameters:

| Parameter | Description |
|------|------|
| `model_provider` | Model provider name, e.g., `vllm`.<br><br>**Type**: `str`<br>**Required**: Yes |
| `base_url` | Default API address of the model provider.<br><br>**Type**: `str`<br>**Required**: No |
| `compatibility_options` | Compatibility options configuration.<br><br>**Type**: `dict`<br>**Required**: No |
| `model_profiles` | Profiles corresponding to the models provided by this model provider.<br><br>**Type**: `dict`<br>**Required**: No |
| `chat_model_cls_name` | Chat model class name, default value is `Chat{model_provider}` (where `{model_provider}` is capitalized).<br><br>**Type**: `str`<br>**Required**: No |

This library builds a chat model class corresponding to a specific provider using the built-in `BaseChatOpenAICompatible` class based on the above parameters passed by the user. This class inherits from `BaseChatOpenAI` of `langchain-openai` and enhances the following capabilities:

- **Support for more formats of reasoning content**: Compared to `ChatOpenAI` which can only output official reasoning content, this class also supports outputting more formats of reasoning content (e.g., `vLLM`).
- **Support for `video` type content_block**: `ChatOpenAI` cannot convert `type=video` `content_block`, this implementation has added support.
- **Dynamically adapt and select the most suitable structured output method**: By default, it can automatically select the optimal structured output method (`function_calling` or `json_schema`) based on the actual support of the model provider.
- **Fine-tune differences through compatibility_options**: By configuring provider compatibility options, resolve support differences for parameters such as `tool_choice`, `response_format`, etc.

!!! warning "Note"
    When using this feature, you must install the standard version of the `langchain-dev-utils` library. Please refer to the installation section for details.

#### Code Example

We take integrating vLLM as an example to show how to use the `create_openai_compatible_model` function to create a chat model integration class.

!!! note "Additional Information"
    vLLM is a commonly used large model inference framework that can deploy large models as OpenAI compatible APIs, such as Qwen3-4B in this example:

    ```bash
    vllm serve Qwen/Qwen3-4B \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3-4b
    ```
    The service address is `http://localhost:8000/v1`.

Use the following code to create a chat model class:

```python
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    base_url="http://localhost:8000/v1",
    chat_model_cls_name="ChatVLLM",
)

model = ChatVLLM(model="qwen3-4b")
print(model.invoke("你好"))
```

It's worth noting that when creating a chat model class, the `base_url` parameter can be omitted. If not passed, this library will read the corresponding environment variable by default. For example:

```bash
export VLLM_API_BASE=http://localhost:8000/v1
```

In this case, the code can omit the `base_url` parameter:

```python
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    chat_model_cls_name="ChatVLLM",
)

model = ChatVLLM(model="qwen3-4b")
print(model.invoke("你好"))
```

**Note**: The prerequisite for the above code to run successfully is that you have set the `VLLM_API_KEY` environment variable. Although vLLM does not require passing an API Key, the chat model class initialization requires an API Key.

!!! info "Tip"
    Naming rules for environment variables of the created chat model class:

    - API address: `${PROVIDER_NAME}_API_BASE` (all uppercase, separated by underscores).

    - API Key: `${PROVIDER_NAME}_API_KEY` (all uppercase, separated by underscores).

#### Compatibility Parameters

`compatibility_options` is a dictionary used to declare the provider's support for some features of the OpenAI API to improve compatibility and stability.

Currently supported configuration items:

| Configuration Item | Description |
|--------|------|
| `supported_tool_choice` | List of supported `tool_choice` strategies.<br><br>**Type**: `list[str]`<br>**Default Value**: `["auto"]` |
| `supported_response_format` | List of supported `response_format` formats (`json_schema`, `json_object`).<br><br>**Type**: `list[str]`<br>**Default Value**: `[]` |
| `reasoning_keep_policy` | Retention policy for the `reasoning_content` field in historical messages.<br><br>**Type**: `str`<br>**Default Value**: `"never"` |
| `include_usage` | Whether to include `usage` information in streaming return results.<br><br>**Type**: `bool`<br>**Default Value**: `True` |

!!! info "Additional Information"
    Since different models from the same model provider have different support for parameters such as `tool_choice` and `response_format`, these four compatibility options are **instance attributes** of the class. Therefore, when creating a chat model class, values can be passed in as global defaults (representing the configuration supported by most models of this provider). If fine-tuning is needed for specific models later, the same-named parameters can be overridden during instantiation.

Detailed introductions to these configuration items are as follows:

??? note "1. supported_tool_choice"
    `tool_choice` is used to control whether and which external tools the large model calls during responses to improve accuracy, reliability, and controllability. Common values include:

    - `"auto"`: The model autonomously decides whether to call tools (default behavior);
    - `"none"`: Prohibit calling tools;
    - `"required"`: Force calling at least one tool;
    - Specify a specific tool (in OpenAI compatible API, specifically `{"type": "function", "function": {"name": "xxx"}}`).

    Different providers support different ranges. To avoid errors, this library defaults `supported_tool_choice` to `["auto"]`, which means when using `bind_tools`, the `tool_choice` parameter can only be passed as `auto`, and other values will be filtered out.

    If you need to support passing other `tool_choice` values, you must configure the supported items. The configuration value is a list of strings, with each optional value being:

    - `"auto"`, `"none"`, `"required"`: Corresponding to standard strategies;
    - `"specific"`: A unique identifier in this library, indicating support for specifying a specific tool.

    For example, vLLM supports all strategies:

    ```python
    from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

    ChatVLLM = create_openai_compatible_model(
        model_provider="vllm",
        chat_model_cls_name="ChatVLLM",
        compatibility_options={
            "supported_tool_choice": ["auto", "required", "none", "specific"]
        },
    )

    model = ChatVLLM(model="qwen3-4b")
    ```

    !!! info "Tip"
        If there are no special requirements, you can keep the default (i.e., `["auto"]`). If business scenarios require the model to **must call a specific tool** or **select one from a given list**, and the model provider supports the corresponding strategy, enable as needed:
        
        1. If you require **at least one tool** to be called and the model provider supports `required`, you can set it to `["required"]` (at the same time, when calling `bind_tools`, you need to explicitly pass `tool_choice="required"`)

        2. If you require **calling a specific** tool and the model provider supports specifying a specific tool call, you can set it to `["specific"]` (in `function_calling` structured output, this configuration is very useful to ensure the model calls the specified structured output tool to ensure the stability of structured output. Because in the `with_structured_output` method, its internal implementation will pass in **a `tool_choice` value that can force calling the specified tool** when calling `bind_tools`, but if `"specific"` is not in `supported_tool_choice`, this parameter will be filtered out. Therefore, if you want to ensure that `tool_choice` can be passed normally, you must add `"specific"` to `supported_tool_choice`.)

        This parameter can be set uniformly when creating, or can be dynamically overridden for a single model during instantiation; it is recommended to declare the `tool_choice` support situation for most models of this provider when creating, and for some models with different support situations, specify separately during instantiation.

??? note "2. supported_response_format"
    Currently, there are three common methods for structured output.

    - `function_calling`: Generate structured output by calling a tool that conforms to a specified schema.
    - `json_schema`: A feature provided by the model provider specifically for generating structured output, in OpenAI compatible API, specifically `response_format={"type": "json_schema", "json_schema": {...}}`.
    - `json_mode`: A feature provided by some providers before they launched `json_schema`, which can generate valid JSON, but the schema must be described in the prompt. In OpenAI compatible API, specifically `response_format={"type": "json_object"}`).

    Among them, `json_schema` is supported by only a few OpenAI compatible API providers (such as `OpenRouter`, `TogetherAI`); `json_mode` has higher support and is compatible with most providers; while `function_calling` is the most universal, as long as the model supports tool calls, it can be used.

    This parameter is used to declare the model provider's support for `response_format`. By default, it is `[]`, which means the model provider supports neither `json_mode` nor `json_schema`. In this case, the `method` parameter in the `with_structured_output` method can only be passed as `function_calling`. If `json_mode` or `json_schema` is passed, it will be automatically converted to `function_calling`. If you want to enable `json_mode` or `json_schema` structured output implementation, you need to explicitly set this parameter.

    For example, if the model deployed by vLLM supports the `json_schema` structured output method, you can declare it during registration:

    ```python
    from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

    ChatVLLM = create_openai_compatible_model(
        model_provider="vllm",
        chat_model_cls_name="ChatVLLM",
        compatibility_options={"supported_response_format": ["json_schema"]},
    )

    model = ChatVLLM(model="qwen3-4b")
    ``` 

    !!! info "Tip"
        Generally, there is no need to configure this. It only needs to be considered when using the `with_structured_output` method. At this time, if the model provider supports `json_schema`, you can consider configuring this parameter (because the stability of `json_schema` structured output is better than `function_calling`). To ensure the stability of structured output. For `json_mode`, because it can only guarantee JSON output, it is generally not necessary to set it. Only when the model does not support tool calls and only supports setting `response_format={"type":"json_object"}`, it is necessary to configure this parameter to include `json_mode`.
        
        Similarly, this parameter can be set uniformly when creating, or can be dynamically overridden for a single model during instantiation; it is recommended to declare the `response_format` support situation for most models of this provider when creating, and for some models with different support situations, specify separately during instantiation.

    !!! warning "Note"
        This parameter currently only affects the `model.with_structured_output` method. For structured output in `create_agent`, if you need to use the `json_schema` implementation, you need to ensure that the corresponding model's `profile` contains the `structured_output` field, and its value is `True`.

??? note "3. reasoning_keep_policy"

    Used to control the retention policy of the `reasoning_content` field in historical messages (messages), mainly adapted to different thinking modes of models from different model providers.

    Supports the following values:

    - `never`: **Do not retain any** reasoning content in historical messages (default);

    - `current`: Only retain the `reasoning_content` field in the **current conversation**;

    - `all`: Retain the `reasoning_content` field in **all conversations**.

    For example:
    For example, the user first asks "How is the weather in New York?", then follows up with "How is the weather in London?", and is currently about to have the second round of conversation, and is about to make the last model call.

    - When the value is `never`

    When the value is `never`, the final messages passed to the model will **not have any** `reasoning_content` field. The final messages received by the model will be:

    ```python
    messages = [
        {"content": "How is the weather in New York?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Cloudy 7~13°C", "role": "tool", "tool_call_id": "..."},
        {"content": "The weather in New York today is cloudy, 7~13°C.", "role": "assistant"},
        {"content": "How is the weather in London?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    - When the value is `current`

    When the value is `current`, only the `reasoning_content` field in the **current conversation** is retained. The final messages received by the model will be:
    ```python
    messages = [
        {"content": "How is the weather in New York?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Cloudy 7~13°C", "role": "tool", "tool_call_id": "..."},
        {"content": "The weather in New York today is cloudy, 7~13°C.", "role": "assistant"},
        {"content": "How is the weather in London?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "Check London weather, need to directly call the weather tool.",  # Only retain reasoning_content of this round of conversation
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
        {"content": "How is the weather in New York?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "Check New York weather, need to directly call the weather tool.",  # Retain reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Cloudy 7~13°C", "role": "tool", "tool_call_id": "..."},
        {
            "content": "The weather in New York today is cloudy, 7~13°C.",
            "reasoning_content": "Directly return New York weather result.",  # Retain reasoning_content
            "role": "assistant",
        },
        {"content": "How is the weather in London?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "Check London weather, need to directly call the weather tool.",  # Retain reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    **Note**: If the current round of conversation does not involve tool calls, the effect of `current` is the same as `never`.

    !!! info "Tip"
        Configure flexibly according to the model provider's requirements for retaining `reasoning_content`:

        - If the provider requires **retaining reasoning content throughout**, set to `all`;  
        - If only required to retain in **this round of tool calls**, set to `current`;  
        - If there are no special requirements, keep the default `never`.

        Similarly, this parameter can be set uniformly when creating, or can be dynamically overridden for a single model during instantiation; **it is generally recommended to specify separately during instantiation**, in which case no need to set when creating.

??? note "4. include_usage"

    `include_usage` is a parameter in the OpenAI compatible API used to control whether to append a message containing token usage information (such as `prompt_tokens` and `completion_tokens`) at the end of the streaming response. Since standard streaming responses do not return usage information by default, enabling this option allows clients to directly obtain complete token consumption data for billing, monitoring, or logging.

    It is usually enabled through `stream_options={"include_usage": true}`. Considering that some model providers do not support this parameter, this library sets it as a compatibility option with a default value of `True`, because most model providers support this parameter. If not supported, it can be explicitly set to `False`.

    !!! info "Tip"
        This parameter generally does not need to be set, just keep the default value. Only when the model provider does not support it, it needs to be set to `False`.

#### model_profiles Parameter Setting

If you want to use the `model.profile` parameter, you must explicitly pass it in when creating.

For example:

```python
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

model_profiles = {
    "qwen3-4b": {
        "max_input_tokens": 131072,
        "max_output_tokens": 8192,
        "image_inputs": False,
        "audio_inputs": False,
        "video_inputs": False,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": True,
        "tool_calling": True,
    }
    # More model profiles can be written here
}

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    base_url="http://localhost:8000/v1",
    model_profiles=model_profiles,
)

model = ChatVLLM(
    model="qwen3-4b",
)
print(model.profile)
```

!!! warning "Note"
    Although the above compatibility configurations have been provided, this library still cannot guarantee 100% compatibility with all OpenAI compatible interfaces. If the model provider already has an official or community integration class, please prioritize using that integration class. If you encounter any compatibility issues, welcome to submit an issue in this library's GitHub repository.

### Creating Embedding Model Class

Similar to the chat model class, you can use `create_openai_compatible_embedding` to create an embedding model class.

#### Example Code
Similarly, we use `create_openai_compatible_embedding` to integrate vLLM's embedding model.

!!! note "Additional Information"  
    vLLM can deploy embedding models and expose OpenAI compatible interfaces, for example:

    ```bash
    vllm serve Qwen/Qwen3-Embedding-4B \
    --task embed \
    --served-model-name qwen3-embedding-4b \
    --host 0.0.0.0 --port 8000
    ```

    The service address is `http://localhost:8000/v1`.

```python
from langchain_dev_utils.embeddings.adapters import create_openai_compatible_embedding

VLLMEmbedding = create_openai_compatible_embedding(
    embedding_provider="vllm",
    base_url="http://localhost:8000/v1",
    embedding_model_cls_name="VLLMEmbedding",
)

embedding = VLLMEmbedding(model="qwen3-embedding-8b")

print(embedding.embed_query("你好"))
```

Similarly, `base_url` can be omitted, in which case you need to set the environment variable `VLLM_API_BASE`.

```bash
export VLLM_API_BASE="http://localhost:8000/v1"
```

The code can omit `base_url`.

```python
from langchain_dev_utils.embeddings.adapters import create_openai_compatible_embedding

VLLMEmbedding = create_openai_compatible_embedding(
    embedding_provider="vllm",
    embedding_model_cls_name="VLLMEmbedding",
)

embedding = VLLMEmbedding(model="qwen3-embedding-8b")

print(embedding.embed_query("你好"))
```

## Using Integration Classes

### Using Chat Model Class

First, we need to create a chat model class. We use the previously created `ChatVLLM` class.

- Supports methods such as `invoke`, `ainvoke`, `stream`, `astream`, etc.

??? example "Regular Call"

    Supports using `invoke` for simple calls:

    ```python
    from langchain_core.messages import HumanMessage

    model = ChatVLLM("qwen3-4b")
    response = model.invoke([HumanMessage("Hello")])
    print(response)
    ```

    Also supports using `ainvoke` for asynchronous calls:

    ```python
    from langchain_core.messages import HumanMessage

    model = ChatVLLM("qwen3-4b")
    response = await model.ainvoke([HumanMessage("Hello")])
    print(response)
    ```

??? example "Streaming Output"

    Supports using `stream` for streaming output:

    ```python
    from langchain_core.messages import HumanMessage

    model = ChatVLLM("qwen3-4b")
    for chunk in model.stream([HumanMessage("Hello")]):
        print(chunk)
    ```

    And using `astream` for asynchronous streaming calls:

    ```python
    from langchain_core.messages import HumanMessage

    model = ChatVLLM("qwen3-4b")
    async for chunk in model.astream([HumanMessage("Hello")]):
        print(chunk)
    ```

- Supports the `bind_tools` method for tool calling.

If the model itself supports tool calling, you can directly use the `bind_tools` method for tool calling:

??? example "Tool Calling"

    ```python
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    import datetime

    @tool
    def get_current_time() -> str:
        """Get current timestamp"""
        return str(datetime.datetime.now().timestamp())

    model = ChatVLLM("qwen3-4b").bind_tools([get_current_time])
    response = model.invoke([HumanMessage("Get current timestamp")])
    print(response)
    ```

- Supports the `with_structured_output` method for structured output.

If the `supported_response_format` parameter of this model class contains `json_schema`, then `with_structured_output` will prioritize using `json_schema` for structured output, otherwise fall back to `function_calling`; if you need `json_mode`, explicitly specify `method="json_mode"` and ensure `json_mode` is included during registration.

??? example "Structured Output"

    ```python
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int

    model = ChatVLLM("qwen3-4b").with_structured_output(User)
    response = model.invoke([HumanMessage("Hello, my name is Zhang San, I'm 25 years old")])
    print(response)
    ```


- Supports passing parameters of `BaseChatOpenAI`, such as `temperature`, `top_p`, `max_tokens`, etc.

In addition, since this class inherits from `BaseChatOpenAI`, it supports passing model parameters of `BaseChatOpenAI`, such as `temperature`, `extra_body`, etc.:

??? example "Passing Model Parameters"

    ```python
    from langchain_core.messages import HumanMessage

    model = ChatVLLM("qwen3-4b", extra_body={"chat_template_kwargs": {"enable_thinking": False}}) # Use extra_body to pass additional parameters, here to disable thinking mode
    response = model.invoke([HumanMessage("Hello")])
    print(response)
    ```


- Supports passing multimodal data

Supports passing multimodal data, you can use OpenAI compatible multimodal data formats or directly use `content_block` in `langchain`.

??? example "Passing Multimodal Data"

    **Passing image type data**:

    ```python
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

    model = ChatVLLM("qwen3-vl-2b")
    response = model.invoke(messages)
    print(response)
    ```

    **Passing video type data**:
    

    ```python
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

    model = ChatVLLM("qwen3-vl-2b")
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


- Supports OpenAI's latest `responses api` (not yet fully guaranteed to be supported, can be used for simple testing, but not for production environments)

This model class also supports OpenAI's latest `responses_api`. However, currently only a few providers support this API style. If your model provider supports this API style, you can pass in the `use_responses_api` parameter as `True`.
    For example, vllm supports `responses_api`, so you can use it like this:

??? example "OpenAI's latest `responses_api`"

    ```python
    from langchain_core.messages import HumanMessage

    model = ChatVLLM("qwen3-4b", use_responses_api=True)
    response = model.invoke([HumanMessage(content="Hello")])
    print(response)
    ```


### Using Embedding Model Class

We use the previously created `VLLMEmbeddings` class to initialize an embedding model instance.

- Vectorize query

??? example "Vectorize Query"

    ```python
    embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
    print(embedding.embed_query("Hello"))
    ```

    **Asynchronous version**

    ```python
    embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
    res = await embedding.aembed_query("Hello")
    print(res)
    ```

- Vectorize string list

??? example "Vectorize String List"

    ```python
    documents = ["Hello", "Hello, I'm Zhang San"]
    embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
    print(embedding.embed_documents(documents))
    ```

    **Asynchronous version**

    ```python
    documents = ["Hello", "Hello, I'm Zhang San"]
    embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
    res = await embedding.aembed_documents(documents)
    print(res)
    ```

**Note**: The chat model class and embedding model class created using this feature support passing any parameters of `BaseChatOpenAI` and `OpenAIEmbeddings`, such as `temperature`, `extra_body`, `dimensions`, etc.

## Integration with Model Management Functionality

This library has seamlessly integrated this functionality into the model management feature. When registering a chat model, simply set `chat_model` to `"openai-compatible"`; when registering an embedding model, set `embeddings_model` to `"openai-compatible"`.

### Chat Model Class Registration

Specific code is as follows:

**Method 1: Explicit Parameter Passing**

```python
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1"
)
```

**Method 2: Through Environment Variables (Recommended for Configuration Management)**

```python
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible"
    # Automatically reads VLLM_API_BASE
)
```

At the same time, the `base_url`, `compatibility_options`, and `model_profiles` parameters in the `create_openai_compatible_model` function also support being passed in. Just pass in the corresponding parameters in the `register_model_provider` function.

### Embedding Model Class Registration

Similar to chat model class registration:

**Method 1: Explicit Parameter Passing**

```python
from langchain_dev_utils.embeddings import register_embeddings_provider

register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)
```

**Method 2: Environment Variables (Recommended)**

```bash
export VLLM_API_BASE=http://localhost:8000/v1
```

```python
from langchain_dev_utils.embeddings import register_embeddings_provider

register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible"
)
```