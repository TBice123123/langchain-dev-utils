# Creation and Usage of Chat Models

## Creating a Chat Model Class

You can use the `create_openai_compatible_model` function to create a chat model integration class. This function accepts the following parameters:

| Parameter | Description |
|------|------|
| `model_provider` | Name of the model provider, e.g., `vllm`. Must start with a letter or number, contain only letters, numbers, and underscores, and be no longer than 20 characters.<br><br>**Type**: `str`<br>**Required**: Yes |
| `base_url` | Default API address for the model provider.<br><br>**Type**: `str`<br>**Required**: No |
| `compatibility_options` | Compatibility options configuration.<br><br>**Type**: `dict`<br>**Required**: No |
| `model_profiles` | Dictionary of profile configurations for models under this provider.<br><br>**Type**: `dict`<br><br>**Required**: No |
| `chat_model_cls_name` | Chat model class name (must conform to Python class naming conventions). Defaults to `Chat{model_provider}` (with the first letter of `{model_provider}` capitalized).<br><br>**Type**: `str`<br>**Required**: No |

The `compatibility_options` is a dictionary used to declare the provider's support for specific OpenAI API features to improve compatibility and stability.

Currently, the following configuration items are supported:

| Configuration Item | Description |
|--------|------|
| `supported_tool_choice` | List of supported `tool_choice` strategies.<br><br>**Type**: `list[str]`<br>**Default**: `["auto"]` |
| `supported_response_format` | List of supported `response_format` formats (`json_schema`, `json_object`).<br><br>**Type**: `list[str]`<br>**Default**: `[]` |
| `reasoning_keep_policy` | Retention policy for the `reasoning_content` field in historical messages.<br><br>**Type**: `str`<br>**Default**: `"never"` |
| `reasoning_field_name` | Field name for reasoning content returned by the provider; generally does not need configuration. Optional values are `reasoning_content` or `reasoning`.<br><br>**Type**: `str`<br>**Default**: `"reasoning_content"` |
| `include_usage` | Whether to include `usage` information in streaming results.<br><br>**Type**: `bool`<br>**Default**: `True` |

!!! info "Supplement"
    Since different models from the same provider may vary in their support for parameters like `tool_choice` and `response_format`, this library treats `supported_tool_choice`, `supported_response_format`, and `reasoning_keep_policy` as **instance attributes** of the class. Default values can be passed when creating the chat model class as a general configuration for the provider; if specific models require fine-tuning, these parameters can be overridden during instantiation.
    
    `reasoning_field_name` and `include_usage` are private attributes of the class and can only be passed via `compatibility_options` when creating or registering the model class.


!!! tip "Tip"
    This library uses the built-in `BaseChatOpenAICompatible` to construct a provider-specific chat model class based on user parameters. This class inherits from `BaseChatOpenAI` in `langchain-openai` and includes the following enhancements:

    **1. Support for extra reasoning fields (reasoning_content / reasoning)**
    ChatOpenAI adheres to the official OpenAI response schema and therefore cannot extract or retain provider-specific fields (e.g., `reasoning_content`, `reasoning`).
    This class extracts and retains `reasoning_content` by default and can be configured via `reasoning_field_name` in `compatibility_options` to extract `reasoning`.

    **2. Dynamic adaptation for structured output methods**
    OpenAICompatibleChatModel selects the best structured output method (`function_calling` or `json_schema`) based on the provider's actual capabilities, utilizing `supported_response_format` in `compatibility_options`.

    **3. Support for relevant parameter configurations**
    This library provides the `compatibility_options` parameter to address discrepancies between provider parameters and the official OpenAI API.
    For example, when different model providers have inconsistent support for `tool_choice`, this can be adapted by setting `supported_tool_choice` in `compatibility_options`.

    **4. Support for `video` type content_block**
    Bridges the capability gap of `ChatOpenAI` regarding `video` type `content_block`.



Use the following code to create a chat model class:

```python hl_lines="4 5 6"
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    base_url="http://localhost:8000/v1",
    chat_model_cls_name="ChatVLLM"
)

model = ChatVLLM(model="qwen2.5-7b")
print(model.invoke("Hello"))
```

When creating a chat model class, the `base_url` parameter can be omitted. If not passed, the library will read the corresponding environment variable by default, for example:

```bash
export VLLM_API_BASE=http://localhost:8000/v1
```

In this case, `base_url` can be omitted in the code:

```python hl_lines="4 5"
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    chat_model_cls_name="ChatVLLM",
)

model = ChatVLLM(model="qwen2.5-7b")
print(model.invoke("Hello"))
```

**Note**: The prerequisite for the above code to run successfully is that the environment variable `VLLM_API_KEY` is configured. Although vLLM itself does not require an API Key, it must be passed during chat model class initialization, so please set this variable first, for example:

```bash
export VLLM_API_KEY=vllm_api_key
```

!!! info "Tip"
    Naming rules for environment variables for created chat model classes (and embedding model classes):

    - API Address: `${PROVIDER_NAME}_API_BASE` (uppercase, underscore separated).

    - API Key: `${PROVIDER_NAME}_API_KEY` (uppercase, underscore separated).


## Using the Chat Model Class

### Standard Invocation

You can perform standard invocation via the `invoke` method, which returns the model response.

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b")
response = model.invoke([HumanMessage("Hello")])
print(response)
```

Asynchronous invocation via `ainvoke` is also supported:

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b")
response = await model.ainvoke([HumanMessage("Hello")])
print(response)
```

### Streaming Invocation

Use the `stream` method for streaming invocation to receive model responses in a stream.

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b")
for chunk in model.stream([HumanMessage("Hello")]):
    print(chunk)
```

And asynchronous streaming invocation via `astream`:

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b")
async for chunk in model.astream([HumanMessage("Hello")]):
    print(chunk)
```

??? note "Streaming Output Options"
    You can append token usage information (`prompt_tokens` and `completion_tokens`) to the end of streaming responses via `stream_options={"include_usage": True}`.
    This library enables this option by default; to disable it, pass the compatibility option `include_usage=False` when creating the model class.

### Tool Calling

If the model supports tool calling, you can use `bind_tools` directly:

```python
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import datetime

@tool
def get_current_time() -> str:
    """Get current timestamp"""
    return str(datetime.datetime.now().timestamp())

model = ChatVLLM(model="qwen2.5-7b").bind_tools([get_current_time])
response = model.invoke([HumanMessage("Get the current timestamp")])
print(response)
```

??? note "Parallel Tool Calling"
    If the model supports parallel tool calling, you can enable it by passing `parallel_tool_calls=True` in `bind_tools` (some providers enable this by default, so no explicit parameter is needed).
    
    For example:

    ```python hl_lines="11"
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool


    @tool
    def get_current_weather(location: str) -> str:
        """Get current weather"""
        return f"The weather in {location} is currently sunny"
    
    model = ChatVLLM(model="qwen2.5-7b").bind_tools(
        [get_current_weather], parallel_tool_calls=True
    )
    response = model.invoke([HumanMessage("Get the weather for Los Angeles and London")])
    print(response)
    ```

??? note "Forcing Tool Calling"

    The `tool_choice` parameter allows you to control whether the model calls a tool and which tool to call, improving accuracy, reliability, and controllability. Common values include:

    - `"auto"`: The model decides whether to call a tool (default behavior);
    - `"none"`: Prohibits tool calling;
    - `"required"`: Forces calling at least one tool;
    - Specifying a specific tool (in OpenAI compatible APIs, specifically `{"type": "function", "function": {"name": "xxx"}}`).

    Different providers have different levels of support for `tool_choice`. To address these differences, this library introduces the compatibility configuration `supported_tool_choice`, which defaults to `["auto"]`. In this case, `tool_choice` passed in `bind_tools` can only be `auto`; other values will be filtered out.

    If you need to support passing other `tool_choice` values, you must configure the supported items. The configuration value is a list of strings, where each string can be:

    - `"auto"`, `"none"`, `"required"`: Correspond to standard strategies;
    - `"specific"`: A library-specific identifier indicating support for specifying a specific tool.

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

    model = ChatVLLM(model="qwen2.5-7b").bind_tools(
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

model = ChatVLLM(model="qwen2.5-7b").with_structured_output(User)
response = model.invoke([HumanMessage("Hello, my name is Zhang San, and I am 25 years old")])
print(response)

```
??? note "Default Structured Output Method"

    There are currently three common structured output methods: `json_schema`, `function_calling`, and `json_mode`. Among them, `json_schema` offers the best performance, so this library's `with_structured_output` prioritizes `json_schema` as the structured output method; only when the provider does not support it will it automatically downgrade to `function_calling`. Different model providers have varying levels of support for structured output. This library declares the methods supported by the provider via the compatibility configuration `supported_response_format`. The default value is `[]`, indicating support for neither `json_schema` nor `json_mode`. In this case, `with_structured_output(method=...)` will strictly use `function_calling`; even if `json_schema` / `json_mode` is passed, it will be converted to `function_calling`. If you want to use a specific structured output method, you need to pass the corresponding parameters explicitly (especially for `json_schema`).

    For example, if a model deployed via vLLM supports the `json_schema` structured output method, you can declare it during registration:

    ```python hl_lines="6"
    from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

    ChatVLLM = create_openai_compatible_model(
        model_provider="vllm",
        chat_model_cls_name="ChatVLLM",
        compatibility_options={"supported_response_format": ["json_schema"]},
    )

    model = ChatVLLM(model="qwen2.5-7b")
    ``` 

    !!! note "Note"
        If `supported_response_format` includes `json_schema`, the `structured_output` field in `model.profile` will automatically be set to `True`. In this case, if no specific structured output strategy is specified when using `create_agent`, `json_schema` will be used by default.

        For example: 
        ```python hl_lines="6"
        from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

        ChatVLLM = create_openai_compatible_model(
            model_provider="vllm",
            base_url="http://localhost:8000/v1",
            compatibility_options={"supported_response_format": ["json_schema"]},
        )

        model = ChatVLLM(model="qwen2.5-7b")
        print(model.profile)
        ```

        The output will be:

        ```
        {'structured_output': True}
        ```


### Passing Extra Parameters

Since this class inherits from `BaseChatOpenAI`, it supports passing `BaseChatOpenAI` model parameters, such as `temperature`, `extra_body`, etc.

For request parameters not defined by the official OpenAI API, you can pass them via `extra_body`.

For example:

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(
    model="qwen2.5-7b",
    extra_body={"top_k": 50},
)
response = model.invoke([HumanMessage("Hello")])
print(response)
```

### Passing Multimodal Data

Passing multimodal data is supported. You can use the OpenAI compatible multimodal data format or directly use `content_block` in LangChain.

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

model = ChatVLLM(model="qwen2.5-vl-7b")
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

model = ChatVLLM(model="qwen2.5-vl-7b")
response = model.invoke(messages)
print(response)
```
   
### Using Reasoning Models

A major feature of the model classes created by this library is further adaptation for reasoning models. For example, integrating the `qwen3-4b` model.

For example:

```python
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model
from langchain_core.messages import HumanMessage

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    base_url="http://localhost:8000/v1",
    chat_model_cls_name="ChatVLLM",
    compatibility_options={
        "reasoning_field_name": "reasoning",
    },
)

model = ChatVLLM(model="qwen3-4b")
response = model.invoke("Why are parrot feathers so colorful?")
reasoning_steps = [b for b in response.content_blocks if b["type"] == "reasoning"]
print(" ".join(step["reasoning"] for step in reasoning_steps))
```
!!! note "Note"
    Since the new version of vLLM returns reasoning content via the `reasoning` field by default, if you are using a reasoning model deployed via vLLM, you must specify the `reasoning_field_name` parameter as `reasoning` when creating the chat model class.

    However, to reuse the existing default parsing logic for `content_blocks`, this library still saves it to `additional_kwargs["reasoning_content"]`.


??? note "Support for Different Reasoning Modes"

    Reasoning modes vary across different models (this is particularly important in Agent development): some require explicitly passing reasoning content in the current call, while others do not. This library provides the `reasoning_keep_policy` compatibility configuration to adapt to these differences.

    This configuration item supports the following values:

    - `never`: Do **not retain any** reasoning content in historical messages (default);

    - `current`: Only retain reasoning content from the **current conversation**;

    - `all`: Retain reasoning content from **all conversations**.

    ```mermaid
    graph LR
        A[reasoning_content retention policy] --> B{Value?};
        B -->|never| C[Contains no<br>reasoning content];
        B -->|current| D[Contains reasoning content<br>from current conversation only<br>Adapts to Interleaved Thinking mode];
        B -->|all| E[Contains reasoning content<br>from all conversations];
        C --> F[Sent to model];
        D --> F;
        E --> F;
    ```
    
    For example, assuming the reasoning content field name is `reasoning_content`. When a user first asks "What is the weather in New York?", then follows up with "What is the weather in London?", currently entering the second round of dialogue and about to make the final model call.

    - When the value is `never`

    The messages passed to the model will **not contain any** `reasoning_content` field. The messages received by the model are:

    ```python
    messages = [
        {"content": "Check weather in New York?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Cloudy 7~13°C", "role": "tool", "tool_call_id": "..."},
        {"content": "New York is cloudy today, 7~13°C.", "role": "assistant"},
        {"content": "Check weather in London?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    - When the value is `current`

    Only the `reasoning_content` field from the **current conversation** is retained. This policy applies to Interleaved Thinking scenarios, where the model alternates between explicit reasoning and tool calls, requiring the retention of reasoning content from the current turn. The messages received by the model are:
    ```python
    messages = [
        {"content": "Check weather in New York?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Cloudy 7~13°C", "role": "tool", "tool_call_id": "..."},
        {"content": "New York is cloudy today, 7~13°C.", "role": "assistant"},
        {"content": "Check weather in London?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "To check London weather, need to call the weather tool directly.",  # Only retain reasoning_content from this round
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    - When the value is `all`

    Retain `reasoning_content` fields from **all** conversations. The messages received by the model are:
    ```python
    messages = [
        {"content": "Check weather in New York?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "To check New York weather, need to call the weather tool directly.",  # Retain reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Cloudy 7~13°C", "role": "tool", "tool_call_id": "..."},
        {
            "content": "New York is cloudy today, 7~13°C.",
            "reasoning_content": "Return New York weather result directly.",  # Retain reasoning_content
            "role": "assistant",
        },
        {"content": "Check weather in London?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "To check London weather, need to call the weather tool directly.",  # Retain reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    **Note**: If the current round of dialogue does not involve tool calls, `current` has the same effect as `never`.

    It is worth noting that while this parameter is a compatibility configuration item, different models from the same provider, or even the same model in different scenarios, may have different requirements for retaining reasoning content. Therefore, it is **recommended to specify it explicitly during instantiation**, and there is no need to assign a value when creating the class.

    For example, taking the GLM-4.7-Flash model, since it supports Interleaved Thinking mode, it generally requires setting `reasoning_keep_policy` to `current` during instantiation to only retain the reasoning content from the current turn. For example:

    ```python hl_lines="3"
    from langchain_core.messages import HumanMessage

    model = ChatVLLM(model="glm-4.7-flash", reasoning_keep_policy="current")
    agent = create_agent(
        model=model,
        tools=[get_current_weather],
    )
    response = agent.invoke({"messages": [HumanMessage(content="Check weather in New York?")]})
    print(response)
    ```
    Meanwhile, the GLM-4.7-Flash model also supports another thinking mode called Preserved Thinking. In this case, all `reasoning_content` fields in historical messages need to be retained, so `reasoning_keep_policy` can be set to `all`. For example:

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
    response = agent.invoke({"messages": [HumanMessage(content="Check weather in New York?")]})
    print(response)
    ```

    !!! note "Note"
       Similarly, the GLM-4.7-Flash model is a reasoning model, and the field for returning reasoning content is `reasoning`. Therefore, it is also necessary to specify the compatibility option `reasoning_field_name` as `reasoning` when creating the chat model class.


### Model profiles

You can access the model's profile via `model.profile`. By default, it returns an empty dictionary.

You can also explicitly pass the `profile` parameter during instantiation to specify the model profile.

For example:
```python
from langchain_core.messages import HumanMessage

custom_profile = {
    "max_input_tokens": 131072,
    "tool_calling": True,
    "structured_output": True,
    # ...
}
model = ChatVLLM(model="qwen2.5-7b", profile=custom_profile)
print(model.profile)
```
Or pass the `profile` parameters for all models of the provider directly during creation.

For example:
```python hl_lines="22"
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

model_profiles = {
     "qwen2.5-7b": {
        "max_input_tokens": 131072,
        "max_output_tokens": 8192,
        "image_inputs": False,
        "audio_inputs": False,
        "video_inputs": False,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": False,
        "tool_calling": True
    },
    # More model profiles can be added here
}

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    base_url="http://localhost:8000/v1",
    model_profiles=model_profiles,
)

model = ChatVLLM(
    model="qwen2.5-7b",
)
print(model.profile)
```

### Support for OpenAI's Latest Responses API

This model class also supports OpenAI's latest `responses` API (parameter name `use_responses_api`). Currently, only a few providers support this style of interface; if your provider supports it, you can enable it via `use_responses_api=True`.

For example, if vLLM supports the `responses` API, you can use it like this:

```python hl_lines="3"
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b", use_responses_api=True)
response = model.invoke([HumanMessage(content="Hello")])
print(response)
```

Currently, the implementation of this feature relies entirely on `BaseChatOpenAI`'s implementation of the `responses` API, so there may be some compatibility issues during use, which will be optimized later based on actual conditions.


!!! warning "Warning"
    This library cannot yet guarantee 100% compatibility with all OpenAI compatible interfaces (although compatibility configurations are used to improve this, differences may still exist). If there is an official or community-maintained integration class for the target model, please prioritize using that. If you encounter compatibility issues, feel free to open an issue on the GitHub repository.

    Taking OpenRouter as an example, while it provides an OpenAI compatible interface, it has multiple compatibility differences; LangChain officially offers [ChatOpenRouter](https://docs.langchain.com/oss/python/integrations/providers/openrouter), so it is recommended to use that class directly to access OpenRouter.

!!! warning "Warning"
    This function uses `pydantic.create_model` at the底层 to create chat model classes, which incurs some performance overhead. Additionally, `create_openai_compatible_model` uses a global dictionary to record the `profiles` of each model provider. To avoid multi-threading concurrency issues, it is recommended to create integration classes during the project startup phase and avoid dynamic creation later.

!!! success "Best Practices"
    When integrating chat model providers with OpenAI compatible APIs, you can directly use `ChatOpenAI` from `langchain-openai` and point `base_url` and `api_key` to your provider service. This approach is simple enough and suitable for relatively simple scenarios (especially when using standard chat models rather than reasoning models).

    However, the following issues exist:

    1. Unable to display the chain of thought (i.e., content returned by `reasoning_content` / `reasoning`) for non-official OpenAI reasoning models.

    2. Does not support `video` type content_block.

    3. Low coverage rate for default structured output strategies.

    When you encounter these differences, you can use the OpenAI compatible integration class provided by this library for adaptation.