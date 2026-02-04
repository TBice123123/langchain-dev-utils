# Creation and Usage of Chat Models

## Creating Chat Model Classes

Use the `create_openai_compatible_model` function to create an integrated chat model class. This function accepts the following parameters:

| Parameter | Description |
|-----------|-------------|
| `model_provider` | Model provider name, e.g., `vllm`. Must start with a letter or number, can only contain letters, numbers, and underscores, with a maximum length of 20 characters.<br><br>**Type**: `str`<br>**Required**: Yes |
| `base_url` | Default API endpoint for the model provider.<br><br>**Type**: `str`<br>**Required**: No |
| `compatibility_options` | Compatibility options configuration.<br><br>**Type**: `dict`<br>**Required**: No |
| `model_profiles` | Profile configuration dictionary for each model of this provider.<br><br>**Type**: `dict`<br>**Required**: No |
| `chat_model_cls_name` | Chat model class name (must comply with Python class naming conventions). Default value is `Chat{model_provider}` (where `{model_provider}` is capitalized).<br><br>**Type**: `str`<br>**Required**: No |

Among them, `compatibility_options` is a dictionary used to declare the provider's support for specific features of the OpenAI API, improving compatibility and stability.

Currently supported configuration items:

| Configuration Item | Description |
|-------------------|-------------|
| `supported_tool_choice` | List of supported `tool_choice` strategies.<br><br>**Type**: `list[str]`<br>**Default**: `["auto"]` |
| `supported_response_format` | List of supported `response_format` formats (`json_schema`, `json_object`).<br><br>**Type**: `list[str]`<br>**Default**: `[]` |
| `reasoning_keep_policy` | Retention policy for the `reasoning_content` field in historical messages.<br><br>**Type**: `str`<br>**Default**: `"never"` |
| `include_usage` | Whether to include `usage` information in streaming responses.<br><br>**Type**: `bool`<br>**Default**: `True` |

!!! info "Supplement"
    Since different models from the same provider may have varying support for parameters like `tool_choice` and `response_format`, these four compatibility options are **instance attributes** of the class. Therefore, when creating a chat model class, you can pass in values as global defaults (representing configurations supported by most models of that provider). If specific models require adjustments, you can override these parameters during instantiation.

!!! tip "Tip"
    This library constructs a provider-specific chat model class using the built-in `BaseChatOpenAICompatible` based on user-provided parameters. This class inherits from `langchain-openai`'s `BaseChatOpenAI` and is enhanced in the following aspects:

    - **Supports more reasoning content formats**: In addition to the official OpenAI format, it also supports reasoning content returned via the `reasoning_content` parameter.
    - **Supports `video` type content_block**: Fills the capability gap of `ChatOpenAI` regarding `video` type `content_block`.
    - **Automatically selects more suitable structured output methods**: Based on the provider's actual support, automatically chooses between `function_calling` and `json_schema` for better solutions.
    - **Fine-grained adaptation of differences via `compatibility_options`**: Configure support differences for parameters like `tool_choice` and `response_format` as needed.

Use the following code to create a chat model class:

```python hl_lines="4 5 6"
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    base_url="http://localhost:8000/v1",
    chat_model_cls_name="ChatVLLM",
)

model = ChatVLLM(model="qwen3-4b")
print(model.invoke("Hello"))
```

When creating a chat model class, the `base_url` parameter can be omitted. If not provided, the library will read the corresponding environment variable by default, for example:

```bash
export VLLM_API_BASE=http://localhost:8000/v1
```

At this point, the code can omit `base_url`:

```python hl_lines="4 5"
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    chat_model_cls_name="ChatVLLM",
)

model = ChatVLLM(model="qwen3-4b")
print(model.invoke("Hello"))
```

**Note**: The above code successfully runs assuming the environment variable `VLLM_API_KEY` is configured. Although vLLM itself does not require an API Key, the chat model class initialization requires one. Therefore, please set this variable first, for example:

```bash
export VLLM_API_KEY=vllm_api_key
```

!!! info "Note"
    The naming rules for environment variables for created chat model classes (embedding model classes follow the same rules):

    - API Base URL: `${PROVIDER_NAME}_API_BASE` (all uppercase, underscore separated).
    - API Key: `${PROVIDER_NAME}_API_KEY` (all uppercase, underscore separated).

## Using the Chat Model Class

### Standard Invocation

Use the `invoke` method for standard invocation, returning the model's response.

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
response = model.invoke([HumanMessage("Hello")])
print(response)
```

Also supports asynchronous invocation via `ainvoke`:

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
response = await model.ainvoke([HumanMessage("Hello")])
print(response)
```

### Streaming Invocation

Use the `stream` method for streaming invocation, for streaming model responses.

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
for chunk in model.stream([HumanMessage("Hello")]):
    print(chunk)
```

And asynchronous streaming invocation via `astream`:

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
async for chunk in model.astream([HumanMessage("Hello")]):
    print(chunk)
```

??? note "Streaming Output Options"
    You can use `stream_options={"include_usage": True}` to append token usage information (`prompt_tokens` and `completion_tokens`) at the end of streaming responses.
    This library enables this option by default; to disable it, you can pass the compatibility option `include_usage=False` when creating the model class or during instantiation.

### Tool Calling

If the model supports tool calling, you can directly use `bind_tools` for tool calling:

```python
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import datetime

@tool
def get_current_time() -> str:
    """Get the current timestamp"""
    return str(datetime.datetime.now().timestamp())

model = ChatVLLM(model="qwen3-4b").bind_tools([get_current_time])
response = model.invoke([HumanMessage("Get the current timestamp")])
print(response)
```
??? note "Parallel Tool Calls"
    If the model supports parallel tool calls, you can pass `parallel_tool_calls=True` in `bind_tools` to enable parallel tool calls (some model providers enable this by default, so explicit passing may not be necessary).

    For example:

    ```python hl_lines="11"
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool


    @tool
    def get_current_weather(location: str) -> str:
        """Get the current weather"""
        return f"The weather in {location} is sunny"
    
    model = ChatVLLM(model="qwen3-4b").bind_tools(
        [get_current_weather], parallel_tool_calls=True
    )
    response = model.invoke([HumanMessage("Get the weather in Los Angeles and London")])
    print(response)
    ```

??? note "Forced Tool Calling"

    The `tool_choice` parameter controls whether the model calls tools and which tool to call in its response, improving accuracy, reliability, and controllability. Common values include:

    - `"auto"`: Model decides whether to call tools (default behavior);
    - `"none"`: Disable tool calling;
    - `"required"`: Force calling at least one tool;
    - Specify a specific tool (in OpenAI-compatible APIs, specifically `{"type": "function", "function": {"name": "xxx"}}`).

    Different providers have varying support ranges for `tool_choice`. To address these differences, this library introduces the compatibility configuration item `supported_tool_choice`, with a default value of `["auto"]`. In this case, the `tool_choice` passed in `bind_tools` can only be `auto`; other values will be filtered out.

    To support other `tool_choice` values, you must configure the supported items. The configuration value is a list of strings, with each string's optional values:

    - `"auto"`, `"none"`, `"required"`: Correspond to standard strategies;
    - `"specific"`: A unique identifier of this library, indicating support for specifying specific tools.

    For example, vLLM supports all strategies:

    ```python hl_lines="6 7 8 12"  
    from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

    ChatVLLM = create_openai_compatible_model(
        model_provider="vllm",
        chat_model_cls_name="ChatVLLM",
        compatibility_options={
            "supported_tool_choice": ["auto", "required", "none", "specific"]
        },
    )

    model = ChatVLLM(model="qwen3-4b").bind_tools(
        [get_current_weather], tool_choice="required"
    )
    ```


### Structured Output

```python
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

model = ChatVLLM(model="qwen3-4b").with_structured_output(User)
response = model.invoke([HumanMessage("Hello, my name is Zhang San, I am 25 years old")])
print(response)
```
??? note "Default Structured Output Method"

    There are currently three common structured output methods: `json_schema`, `function_calling`, `json_mode`. Among them, `json_schema` yields the best results, so this library's `with_structured_output` prioritizes using `json_schema` as the structured output method; when the provider does not support it, it automatically falls back to `function_calling`. Different model providers have varying levels of support for structured output. This library declares the supported structured output methods via the compatibility configuration item `supported_response_format`. The default value is `[]`, indicating neither `json_schema` nor `json_mode` is supported. In this case, `with_structured_output(method=...)` will consistently use `function_calling`; even if `json_schema` / `json_mode` is passed, it will be automatically converted to `function_calling`. If you want to use the corresponding structured output method, you need to explicitly pass the relevant parameters (especially for `json_schema`).

    For example, if a model deployed via vLLM supports the `json_schema` structured output method, you can declare it during registration:

    ```python hl_lines="6"
    from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

    ChatVLLM = create_openai_compatible_model(
        model_provider="vllm",
        chat_model_cls_name="ChatVLLM",
        compatibility_options={"supported_response_format": ["json_schema"]},
    )

    model = ChatVLLM(model="qwen3-4b")
    ``` 

    !!! note "Note"
        If `supported_response_format` includes `json_schema`, the `structured_output` field in `model.profile` will automatically be set to `True`. In this case, when using `create_agent` for structured output without specifying a specific structured output strategy, `json_schema` will be used as the default structured output strategy.

        For example: 
        ```python hl_lines="6"
        from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

        ChatVLLM = create_openai_compatible_model(
            model_provider="vllm",
            base_url="http://localhost:8000/v1",
            compatibility_options={"supported_response_format": ["json_schema"]},
        )

        model = ChatVLLM(model="qwen3-4b")
        print(model.profile)
        ```

        Output result:

        ```
        {'structured_output': True}
        ```


### Passing Additional Parameters

Since this class inherits from `BaseChatOpenAI`, it supports passing model parameters of `BaseChatOpenAI`, such as `temperature`, `extra_body`, etc.

For example, using `extra_body` to pass additional parameters (here, disabling thinking mode):

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(
    model="qwen3-4b",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
response = model.invoke([HumanMessage("Hello")])
print(response)
```

### Passing Multimodal Data

Supports passing multimodal data. You can use the OpenAI-compatible multimodal data format or directly use `content_block` from LangChain.

Passing image data:

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

model = ChatVLLM(model="qwen3-vl-2b")
response = model.invoke(messages)
print(response)
```
Passing video data:

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

model = ChatVLLM(model="qwen3-vl-2b")
response = model.invoke(messages)
print(response)
```
   
### Using Reasoning Models

A major feature of the model classes created by this library is further adaptation to more reasoning models.

For example:

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
response = model.invoke("Why are parrot feathers so colorful?")
reasoning_steps = [b for b in response.content_blocks if b["type"] == "reasoning"]
print(" ".join(step["reasoning"] for step in reasoning_steps))
```

??? note "Support for Different Reasoning Modes"

    Different models have varying reasoning modes (especially important in Agent development): some require explicitly passing the `reasoning_content` field in the current call, while others do not. This library provides the `reasoning_keep_policy` compatibility configuration to adapt to these differences.

    This configuration item supports the following values:

    - `never`: **Do not retain any** reasoning content in historical messages (default);

    - `current`: Only retain the `reasoning_content` field from the **current conversation**;

    - `all`: Retain the `reasoning_content` field from **all conversations**.

    ```mermaid
    graph LR
        A[reasoning_content Retention Policy] --> B{Value?};
        B -->|never| C[Contains no<br>reasoning_content];
        B -->|current| D[Only contains current conversation's<br>reasoning_content<br>Adapts to interleaved thinking mode];
        B -->|all| E[Contains all conversations'<br>reasoning_content];
        C --> F[Send to model];
        D --> F;
        E --> F;
    ```

    For example, the user first asks "What's the weather in New York?", then follows up with "What's the weather in London?". We are currently in the second round of conversation and about to make the final model call.

    - When the value is `never`

    There will be **no** `reasoning_content` fields in the messages passed to the model. The messages the model receives are:

    ```python
    messages = [
        {"content": "Check the weather in New York?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Cloudy 7~13°C", "role": "tool", "tool_call_id": "..."},
        {"content": "The weather in New York today is cloudy, 7~13°C.", "role": "assistant"},
        {"content": "Check the weather in London?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    - When the value is `current`

    Only retain the `reasoning_content` field from the **current conversation**. This strategy is suitable for Interleaved Thinking scenarios, where the model alternates between explicit reasoning and tool calls, requiring retention of reasoning content from the current round. The messages the model receives are:
    ```python
    messages = [
        {"content": "Check the weather in New York?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Cloudy 7~13°C", "role": "tool", "tool_call_id": "..."},
        {"content": "The weather in New York today is cloudy, 7~13°C.", "role": "assistant"},
        {"content": "Check the weather in London?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "Check London weather, need to directly call weather tool.",  # Only retain current round's reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    - When the value is `all`

    Retain the `reasoning_content` field from **all** conversations. The messages the model receives are:
    ```python
    messages = [
        {"content": "Check the weather in New York?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "Check New York weather, need to directly call weather tool.",  # Retain reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Cloudy 7~13°C", "role": "tool", "tool_call_id": "..."},
        {
            "content": "The weather in New York today is cloudy, 7~13°C.",
            "reasoning_content": "Directly return New York weather result.",  # Retain reasoning_content
            "role": "assistant",
        },
        {"content": "Check the weather in London?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "Check London weather, need to directly call weather tool.",  # Retain reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    **Note**: If the current round does not involve tool calls, `current` and `never` have the same effect.

    It's worth noting that although this parameter is a compatibility configuration item, different models from the same provider, or even the same model in different scenarios, may require different `reasoning_content` retention policies. Therefore, **it is recommended to explicitly specify it during instantiation**, and it's not necessary to assign a value when creating the class.

    For example, for the GLM-4.7-Flash model, since it supports Interleaved Thinking mode, you generally need to set `reasoning_keep_policy` to `current` during instantiation to retain only the current round's `reasoning_content`. For example:

    ```python hl_lines="3"
    from langchain_core.messages import HumanMessage

    model = ChatVLLM(model="glm-4.7-flash", reasoning_keep_policy="current")
    agent = create_agent(
        model=model,
        tools=[get_current_weather],
    )
    response = agent.invoke({"messages": [HumanMessage(content="Check the weather in New York?")]})
    print(response)
    ```
    Additionally, the GLM-4.7-Flash model also supports another thinking mode called Preserved Thinking. This requires retaining all `reasoning_content` fields from historical messages, so you can set `reasoning_keep_policy` to `all`. For example:

    ```python hl_lines="5"
    from langchain_core.messages import HumanMessage

    model = ChatVLLM(
        model="glm-4.7-flash",
        reasoning_keep_policy="all",
        extra_body={"chat_template_kwargs": {"clear_thinking": False}},
    )

    agent = create_agent(
        model=model,
        tools=[get_current_weather],
    )
    response = agent.invoke({"messages": [HumanMessage(content="Check the weather in New York?")]})
    print(response)
    ```


### Model Profiles

You can get the model's profile via `model.profile`. By default, it returns an empty dictionary.

You can also explicitly pass a `profile` parameter during instantiation to specify the model profile.

For example:
```python
from langchain_core.messages import HumanMessage

custom_profile = {
    "max_input_tokens": 100_000,
    "tool_calling": True,
    "structured_output": True,
    # ...
}
model = ChatVLLM(model="qwen3-4b", profile=custom_profile)
print(model.profile)
```
Or directly pass the `profile` parameter for all models of the provider during creation.

For example:
```python hl_lines="22"
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
    # More model profiles can be added here
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

### Support for OpenAI's Latest Responses API

This model class also supports OpenAI's latest `responses` API (parameter name `use_responses_api`). Currently, only a few providers support this style of interface; if your provider supports it, you can enable it via `use_responses_api=True`.

For example, if vLLM supports the `responses` API, you can use it like this:

```python hl_lines="3"
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b", use_responses_api=True)
response = model.invoke([HumanMessage(content="Hello")])
print(response)
```

Currently, the implementation of this feature relies entirely on `BaseChatOpenAI`'s implementation of the `responses` API, so there may be certain compatibility issues during use. Subsequent optimizations will be made based on actual circumstances.

!!! warning "Note"
    This library currently cannot guarantee 100% compatibility with all OpenAI-compatible interfaces (although compatibility configurations can improve compatibility). If the model provider has an official or community integration class, please prioritize using that integration class. If you encounter any compatibility issues, feel free to submit an issue on this library's GitHub repository.

!!! warning "Note"
    This function uses `pydantic.create_model` under the hood to create chat model classes, which incurs some performance overhead. Additionally, `create_openai_compatible_model` uses a global dictionary to record the `profiles` of each model provider. To avoid multi-threading concurrency issues, it is recommended to create integration classes during project startup and avoid dynamic creation afterward.

!!! success "Best Practice"
    When connecting to an OpenAI-compatible API chat model provider, you can directly use `langchain-openai`'s `ChatOpenAI` and point `base_url` and `api_key` to your provider's service. This method is simple enough and suitable for relatively simple scenarios (especially when using ordinary chat models rather than reasoning models).

    However, it may have the following issues:

    1. Cannot display the chain of thought (i.e., content returned by `reasoning_content`) of non-OpenAI official reasoning models.

    2. Does not support `video` type content_block.

    3. Lower coverage for default structured output strategies.

    When you encounter the above differences, you can use the OpenAI-compatible integration classes provided by this library for adaptation.