# Creating and Using Chat Models

## Creating a Chat Model Class

Use the `create_openai_compatible_model` function to create a chat model integration class. This function accepts the following parameters:

| Parameter | Description |
|------|------|
| `model_provider` | The name of the model provider, e.g., `vllm`. Must start with a letter or number, contain only letters, numbers, and underscores, and be no longer than 20 characters.<br><br>**Type**: `str`<br>**Required**: Yes |
| `base_url` | The default API address for the model provider.<br><br>**Type**: `str`<br>**Required**: No |
| `compatibility_options` | Compatibility options configuration.<br><br>**Type**: `dict`<br>**Required**: No |
| `model_profiles` | A dictionary of profile configurations for various models under this provider.<br><br>**Type**: `dict`<br>**Required**: No |
| `chat_model_cls_name` | The chat model class name (must conform to Python class naming conventions). The default value is `Chat{model_provider}` (where the first letter of `{model_provider}` is capitalized).<br><br>**Type**: `str`<br>**Required**: No |

`compatibility_options` is a dictionary used to declare the support status and configuration methods for various features of the provider to improve compatibility and stability. These features include the level of support for official OpenAI API features (such as `tool_choice`, `response_format`), as well as unofficial extended features (such as reasoning fields `reasoning_content` / `reasoning`).

Currently, the following configuration items are supported:

| Configuration Item | Description |
|--------|------|
| `supported_tool_choice` | A list of supported `tool_choice` strategies.<br><br>**Type**: `list[str]`<br>**Default Value**: `["auto"]` |
| `supported_response_format` | A list of supported `response_format` formats (`json_schema`, `json_object`).<br><br>**Type**: `list[str]`<br>**Default Value**: `[]` |
| `reasoning_keep_policy` | The retention policy for the `reasoning_content` field in historical messages.<br><br>**Type**: `str`<br>**Default Value**: `"never"` |
| `reasoning_field_name` | The field name the provider uses to return reasoning content; generally does not need configuration. Options are `reasoning_content` or `reasoning`.<br><br>**Type**: `str`<br>**Default Value**: `"reasoning_content"` |
| `include_usage` | Whether to include `usage` information in streaming results.<br><br>**Type**: `bool`<br>**Default Value**: `True` |

!!! info "Additional Notes"
    Different models from the same provider may have different levels of support for parameters like `tool_choice` and `response_format`. Therefore, this library designs `supported_tool_choice`, `supported_response_format`, and `reasoning_keep_policy` as **instance attributes** of the class. You can pass these parameters when creating the chat model class to set the default support status for most models of that provider; if specific models require fine-tuning later, they can be overridden during instantiation.

    `reasoning_field_name` and `include_usage` are private attributes of the class (implemented via [pydantic's PrivateAttr](https://docs.pydantic.dev/latest/concepts/models/#private-model-attributes)). They can only be passed via `compatibility_options` when creating or registering the model class and cannot be overridden during instantiation.


!!! tip "Feature Enhancements"
    Based on user-provided parameters, this library uses the built-in `BaseChatOpenAICompatible` to build a chat model class oriented towards specific providers. This class inherits from `BaseChatOpenAI` in `langchain-openai` and includes the following enhancements:

    **1. Support for Additional Reasoning Fields (reasoning_content / reasoning)**

    `ChatOpenAI` follows the official OpenAI response format and cannot extract or retain provider-specific fields (e.g., `reasoning_content`, `reasoning`). This class extracts and retains `reasoning_content` by default and can be configured via `reasoning_field_name` in `compatibility_options` to extract `reasoning`.

    **2. Dynamic Adaptation of Structured Output Methods**

    The chat model class created by this library dynamically selects the best structured output method (`function_calling` or `json_schema`) based on the provider's actual capabilities, utilizing `supported_response_format` in `compatibility_options`.

    **3. Support for Parameter Differentiation**

    For cases where certain parameters differ from the official OpenAI API, this library provides the `compatibility_options` parameter for adaptation. For example, when different model providers have inconsistent support for `tool_choice`, adaptation can be achieved by setting `supported_tool_choice`.

    **4. Support for `video` Type content_block**

    Bridges the capability gap of `ChatOpenAI` regarding video type `content_block`.



Example code for creating a chat model class:

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

The `base_url` parameter can be omitted when creating the chat model class. If not passed, the library will read the corresponding environment variable, for example:

```bash
export VLLM_API_BASE=http://localhost:8000/v1
```

In this case, the code can omit `base_url`:

```python hl_lines="4 5"
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    chat_model_cls_name="ChatVLLM",
)

model = ChatVLLM(model="qwen2.5-7b")
print(model.invoke("Hello"))
```

**Note**: The prerequisite for the above code to run successfully is that the environment variable `VLLM_API_KEY` has been configured. Although vLLM itself does not require an API Key, it is needed when initializing the chat model class, so please set this variable first, for example:

```bash
export VLLM_API_KEY=vllm_api_key
```

!!! info "Environment Variable Naming Rules"
    Environment variable naming rules for created chat model classes (embedding model classes follow the same rule):

    - API Address: `${PROVIDER_NAME}_API_BASE` (all uppercase, underscore separated)

    - API Key: `${PROVIDER_NAME}_API_KEY` (all uppercase, underscore separated)


## Using the Chat Model Class

### Standard Invocation

Use the `invoke` method for standard invocation, which returns the model response:

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b")
response = model.invoke([HumanMessage("Hello")])
print(response)
```

It also supports `ainvoke` for asynchronous invocation:

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b")
response = await model.ainvoke([HumanMessage("Hello")])
print(response)
```

### Streaming Invocation

Use the `stream` method for streaming invocation:

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b")
for chunk in model.stream([HumanMessage("Hello")]):
    print(chunk)
```

And use `astream` for asynchronous streaming invocation:

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b")
async for chunk in model.astream([HumanMessage("Hello")]):
    print(chunk)
```

??? note "Streaming Output Options"
    You can append token usage information (`prompt_tokens` and `completion_tokens`) at the end of the streaming response via `stream_options={"include_usage": True}`. This library enables this option by default; to disable it, pass the compatibility option `include_usage=False` when creating the model class.

### Tool Calling

If the model supports tool calling, you can directly use `bind_tools` for tool invocation:

```python
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import datetime

@tool
def get_current_time() -> str:
    """Get the current timestamp"""
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
        """Get the current weather"""
        return f"The weather in {location} is currently sunny"

    model = ChatVLLM(model="qwen2.5-7b").bind_tools(
        [get_current_weather], parallel_tool_calls=True
    )
    response = model.invoke([HumanMessage("Get the weather for Los Angeles and London")])
    print(response)
    ```

??? note "Forcing Tool Calling"

    Through the `tool_choice` parameter, you can control whether the model calls a tool and which tool to call when responding, to improve accuracy and controllability. Common values:

    - `"auto"`: The model decides autonomously whether to call a tool (default behavior)
    - `"none"`: Prohibits tool calling
    - `"required"`: Forces calling at least one tool
    - Specify a specific tool (in OpenAI compatible APIs, the format is `{"type": "function", "function": {"name": "xxx"}}`)

    Different providers have different ranges of support for `tool_choice`. To resolve these differences, this library introduces the compatibility configuration item `supported_tool_choice`, which defaults to `["auto"]`. In this case, the `tool_choice` passed in `bind_tools` can only be `auto`; other values will be filtered out.

    If you need to support other `tool_choice` values, you must configure the supported items. The configuration value is a list of strings, where each string can be:

    - `"auto"`, `"none"`, `"required"`: Correspond to standard strategies
    - `"specific"`: A unique identifier in this library, indicating support for specifying a specific tool

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

    Currently, there are three common structured output methods: `json_schema`, `function_calling`, and `json_mode`. Among them, `json_schema` is the most effective, so this library's `with_structured_output` prioritizes using `json_schema` as the structured output method; when the provider does not support it, it automatically falls back to `function_calling`.

    Different model providers have different levels of support for structured output. This library declares the structured output methods supported by the provider through the compatibility configuration item `supported_response_format`. The default value is `[]`, indicating support for neither `json_schema` nor `json_mode`. In this case, `with_structured_output(method=...)` will fixedly use `function_calling`; even passing `json_schema` / `json_mode` will be automatically converted to `function_calling`. If you want to use the corresponding structured output method, you need to pass the relevant parameters explicitly (especially `json_schema`).

    For example, models deployed via vLLM support the `json_schema` structured output method, which can be declared during registration:

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
        If `supported_response_format` contains `json_schema`, the `structured_output` field in `model.profile` will automatically be set to `True`. At this point, when using `create_agent` for structured output, if no specific structured output strategy is specified, `json_schema` will be used by default.

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

        The output result is:

        ```
        {'structured_output': True}
        ```


### Passing Extra Parameters

Since this class inherits from `BaseChatOpenAI`, it supports passing model parameters from `BaseChatOpenAI`, such as `temperature`, `extra_body`, etc.

For request parameters not defined in the official OpenAI API, they can be passed via `extra_body`:

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

Supports passing multimodal data. You can use OpenAI compatible multimodal data formats or directly use `content_block` in LangChain.

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

The model class created by this library is deeply adapted for reasoning models. Taking the deployment of the `qwen3-4b` reasoning model using vLLM as an example:

```python hl_lines="9"
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
response = model.invoke("Why are parrot feathers so brightly colored?")
reasoning_steps = [b for b in response.content_blocks if b["type"] == "reasoning"]
print(" ".join(step["reasoning"] for step in reasoning_steps))
```

!!! note "Note"
    Since the new version of vLLM returns reasoning content in the `reasoning` field by default, if you use vLLM to deploy a reasoning model, you must specify the `reasoning_field_name` parameter as `reasoning` when creating the chat model class.

    However, to reuse the existing default parsing logic for `content_blocks`, this library still saves it to `additional_kwargs["reasoning_content"]`.


??? note "Support for Different Reasoning Modes"

    Different models have different reasoning modes (this is particularly important in Agent development): some require explicit transmission of reasoning content in the current call, while others do not. This library provides the `reasoning_keep_policy` compatibility configuration to adapt to these differences.

    This configuration item supports the following values:

    - `never`: Do **not retain any** reasoning content in historical messages (default value)
    - `current`: Retain reasoning content only from the **current conversation**
    - `all`: Retain reasoning content from **all conversations**

    ```mermaid
    graph LR
        A[reasoning_content retention policy] --> B{Value?};
        B -->|never| C[Does not contain any<br>reasoning content];
        B -->|current| D[Contains reasoning content<br>from the current conversation only<br>Adapted for interleaved thinking mode];
        B -->|all| E[Contains reasoning content<br>from all conversations];
        C --> F[Sent to model];
        D --> F;
        E --> F;
    ```

    For example, assuming the field name for reasoning content is `reasoning_content`, the user first asks "How is the weather in New York?", then follows up with "How is the weather in London?", and the last model call of the second round of conversation is currently taking place.

    - When the value is `never`

    The messages finally passed to the model will **not have any** `reasoning_content` fields. The messages received by the model are:

    ```python
    messages = [
        {"content": "Check the weather in New York?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Cloudy 7~13°C", "role": "tool", "tool_call_id": "..."},
        {"content": "New York is cloudy today, 7~13°C.", "role": "assistant"},
        {"content": "Check the weather in London?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    - When the value is `current`

    Only the `reasoning_content` field from the **current conversation** is retained. This policy applies to Interleaved Thinking scenarios, where the model alternates between explicit reasoning and tool calls, requiring the retention of reasoning content from the current turn. The messages received by the model are:

    ```python
    messages = [
        {"content": "Check the weather in New York?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Cloudy 7~13°C", "role": "tool", "tool_call_id": "..."},
        {"content": "New York is cloudy today, 7~13°C.", "role": "assistant"},
        {"content": "Check the weather in London?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "To check London weather, I need to call the weather tool directly.",  # Only retain reasoning_content from this round
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    - When the value is `all`

    The `reasoning_content` field from **all** conversations is retained. The messages received by the model are:

    ```python
    messages = [
        {"content": "Check the weather in New York?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "To check New York weather, I need to call the weather tool directly.",  # Retain reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Cloudy 7~13°C", "role": "tool", "tool_call_id": "..."},
        {
            "content": "New York is cloudy today, 7~13°C.",
            "reasoning_content": "Directly return the New York weather result.",  # Retain reasoning_content
            "role": "assistant",
        },
        {"content": "Check the weather in London?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "To check London weather, I need to call the weather tool directly.",  # Retain reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Rainy, 14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    **Note**: If the current round of conversation does not involve tool calls, `current` has the same effect as `never`.

    Although this parameter belongs to the compatibility configuration, different models from the same provider, or even the same model in different scenarios, may have different requirements for the retention policy of reasoning content. Therefore, it is **recommended to specify it explicitly during instantiation**; there is no need to assign it when creating the class.

    Currently, the vast majority of the latest generation of reasoning models (especially open-source models) adopt the "Interleaved Thinking" mode. Taking GLM-4.7-Flash as an example, when enabling this mode, you need to set `reasoning_keep_policy` to `current` during instantiation to only retain reasoning content from the current turn. Example:

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

    The GLM-4.7-Flash model also supports another thinking mode—Preserved Thinking. At this time, all `reasoning_content` fields in historical messages need to be retained, which can be done by setting `reasoning_keep_policy` to `all`. For example:

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

    !!! note "Note"

        When GLM-4.7-Flash is deployed using vLLM as a reasoning model, if you want to return reasoning content to achieve interleaved thinking, the corresponding field is `reasoning` instead of `reasoning_content`. Therefore, you also need to specify the compatibility option `reasoning_field_name` as `reasoning` when creating the chat model class.


### Model profiles

You can get the model's profile via `model.profile`. By default, it returns an empty dictionary.

You can also explicitly pass the `profile` parameter during instantiation to specify the model profile:

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

Or pass the `profile` parameters for all models of the provider directly during creation:

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

For example, vLLM supports the `responses` API:

```python hl_lines="3"
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b", use_responses_api=True)
response = model.invoke([HumanMessage(content="Hello")])
print(response)
```

Currently, the implementation of this feature relies entirely on `BaseChatOpenAI`'s implementation of the `responses` API. There may be some compatibility issues during use, and optimizations will be made later based on the actual situation.


!!! warning "Compatibility Statement"
    This library cannot yet guarantee 100% compatibility with all OpenAI compatible interfaces (although compatibility can be improved as much as possible through compatibility configurations, differences may still exist). If there is an official or community-maintained integration class for the target model, please use it first. If you encounter compatibility issues, feel free to open an issue in the GitHub repository.

    Taking OpenRouter as an example, although it provides an OpenAI compatible interface, there are multiple compatibility differences; LangChain official has launched [ChatOpenRouter](https://docs.langchain.com/oss/python/integrations/providers/openrouter), it is recommended to use that class directly to access OpenRouter.

!!! warning "Performance Notes"
    This function uses [pydantic's create_model](https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation) at the underlying level to create chat model classes, which incurs a certain performance overhead. In addition, `create_openai_compatible_model` uses a global dictionary to record the `profiles` of each model provider. To avoid multi-threading concurrency issues, it is recommended to create integration classes during the project startup phase and avoid dynamic creation later.

!!! success "Best Practices"
    When integrating chat model providers with OpenAI compatible APIs, you can directly use `ChatOpenAI` from `langchain-openai` and point to your provider service via `base_url` and `api_key`. This method is simple and direct, suitable for simple scenarios (especially when using standard chat models rather than reasoning models).

    However, there are the following issues:

    1. Cannot display the chain of thought of unofficial OpenAI reasoning models (i.e., content returned by `reasoning_content` / `reasoning`)

    2. Does not support `video` type content_block

    3. Low coverage rate of default structured output strategies

    When you encounter the above problems, you can use the OpenAI compatible integration class provided by this library for adaptation.