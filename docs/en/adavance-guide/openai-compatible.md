# OpenAI Compatible API Model Provider Integration

!!! warning "Prerequisites"
    When using this feature, the standard version of the `langchain-dev-utils` library must be installed. Please refer to the installation section for details.

## Overview

Many model providers offer **OpenAI Compatible API** services, such as [vLLM](https://github.com/vllm-project/vllm), [OpenRouter](https://openrouter.ai/), and [Together AI](https://www.together.ai/). This library provides an OpenAI compatible API integration solution covering both chat models and embedding models. It is particularly suitable for scenarios where "the provider offers an OpenAI compatible API but there is no corresponding LangChain integration yet."

This library provides two utility functions for creating chat model integration classes and embedding model integration classes:

| Function Name | Description |
|---------------|-------------|
| `create_openai_compatible_model` | Creates a chat model integration class |
| `create_openai_compatible_embedding` | Creates an embedding model integration class |


!!! tip "Note"
    The initial inspiration for the two utility functions provided by this library comes from the JavaScript ecosystem's [@ai-sdk/openai-compatible](https://ai-sdk.dev/providers/openai-compatible-providers).

The following example demonstrates how to use this feature by integrating with [vLLM](https://github.com/vllm-project/vllm).

??? note "vLLM Introduction"
    vLLM is a popular LLM inference framework that can deploy large models as an OpenAI compatible API.

    For example:

    Deploying a normal text model such as **Qwen3-4B**:

    ```bash
    vllm serve Qwen/Qwen3-4B \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3-4b
    ```

    Deploying a model that requires special handling, such as **GLM-4.7-Flash**:

    ```bash
    vllm serve zai-org/GLM-4.7-Flash \
     --tensor-parallel-size 4 \
     --speculative-config.method mtp \
     --speculative-config.num_speculative_tokens 1 \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --served-model-name glm-4.7-flash
    ```

    Deploying a multimodal model such as **Qwen3-VL-2B-Instruct**:

    ```bash
    vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3-vl-2b
    ```

    Deploying an embedding model such as **Qwen3-Embedding-4B**:

    ```bash
    vllm serve Qwen/Qwen3-Embedding-4B \
    --task embed \
    --served-model-name qwen3-embedding-4b \
    --host 0.0.0.0 --port 8000
    ```
    The service address is `http://localhost:8000/v1`.


## Creation and Usage of Chat Models

### Creating a Chat Model Class

You can use the `create_openai_compatible_model` function to create a chat model integration class. This function accepts the following parameters:

| Parameter | Description |
|-----------|-------------|
| `model_provider` | Model provider name, e.g., `vllm`. Must start with a letter or number, contain only letters, numbers, and underscores, and be no longer than 20 characters.<br><br>**Type**: `str`<br>**Required**: Yes |
| `base_url` | Default API address for the model provider.<br><br>**Type**: `str`<br>**Required**: No |
| `compatibility_options` | Compatibility option configuration.<br><br>**Type**: `dict`<br>**Required**: No |
| `model_profiles` | Dictionary of profile configurations for the provider's models.<br><br>**Type**: `dict`<br>**Required**: No |
| `chat_model_cls_name` | Chat model class name (must conform to Python class naming conventions). Default is `Chat{model_provider}` (with `{model_provider}` capitalized).<br><br>**Type**: `str`<br>**Required**: No |

Among these, `compatibility_options` is a dictionary used to declare the provider's support for certain OpenAI API features to improve compatibility and stability.

Currently, the following configuration items are supported:

| Configuration Item | Description |
|--------------------|-------------|
| `supported_tool_choice` | List of supported `tool_choice` strategies.<br><br>**Type**: `list[str]`<br>**Default**: `["auto"]` |
| `supported_response_format` | List of supported `response_format` formats (`json_schema`, `json_object`).<br><br>**Type**: `list[str]`<br>**Default**: `[]` |
| `reasoning_keep_policy` | Retention policy for the `reasoning_content` field in historical messages.<br><br>**Type**: `str`<br>**Default**: `"never"` |
| `include_usage` | Whether to include `usage` information in streaming results.<br><br>**Type**: `bool`<br>**Default**: `True` |

!!! info "Supplement"
    Since different models from the same provider may have varying support for parameters like `tool_choice` and `response_format`, these four compatibility options are **instance attributes** of the class. Therefore, when creating the chat model class, you can pass values as global defaults (representing the configuration supported by most models of that provider). If you need to fine-tune for a specific model later, you can override the parameters with the same name during instantiation.


!!! tip "Hint"
    Based on the parameters provided by the user, this library uses the built-in `BaseChatOpenAICompatible` to construct a provider-specific chat model class. This class inherits from `BaseChatOpenAI` in `langchain-openai` and includes the following enhancements:

    - **Support for more reasoning content formats**: In addition to the official OpenAI format, it also supports the reasoning content format returned via the `reasoning_content` parameter.
    - **Support for `video` type content_block**: Fills the capability gap of `ChatOpenAI` regarding video type `content_block`.
    - **Automatic selection of optimal structured output method**: Automatically selects the better solution between `function_calling` and `json_schema` based on actual provider support.
    - **Fine adaptation of differences via `compatibility_options`**: Configure support differences for parameters like `tool_choice` and `response_format` as needed.


Use the following code to create a chat model class:

```python
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    base_url="http://localhost:8000/v1",
    chat_model_cls_name="ChatVLLM",
)

model = ChatVLLM(model="qwen3-4b")
print(model.invoke("Hello"))
```

When creating the chat model class, the `base_url` parameter can be omitted. If not passed, the library will default to reading the corresponding environment variable, for example:

```bash
export VLLM_API_BASE=http://localhost:8000/v1
```

In this case, the code can omit `base_url`:

```python
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    chat_model_cls_name="ChatVLLM",
)

model = ChatVLLM(model="qwen3-4b")
print(model.invoke("Hello"))
```

**Note**: The prerequisite for the above code to run successfully is that the environment variable `VLLM_API_KEY` has been configured. Although vLLM itself does not require an API Key, the chat model class requires one during initialization. Therefore, please set this variable first, for example:

```bash
export VLLM_API_KEY=vllm_api_key
```

!!! info "Hint"
    The naming rules for environment variables of the created chat model class (embedding model classes follow this rule as well):

    - API Address: `${PROVIDER_NAME}_API_BASE` (uppercase, separated by underscores).

    - API Key: `${PROVIDER_NAME}_API_KEY` (uppercase, separated by underscores).


### Using the Chat Model Class

#### Basic Invocation

You can perform basic invocation via the `invoke` method, which returns the model response.

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
response = model.invoke([HumanMessage("Hello")])
print(response)
```

It also supports `ainvoke` for asynchronous invocation:

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
response = await model.ainvoke([HumanMessage("Hello")])
print(response)
```
#### Streaming Invocation

You can perform streaming invocation via the `stream` method, used to stream the model response back.

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
for chunk in model.stream([HumanMessage("Hello")]):
    print(chunk)
```

And asynchronous streaming via `astream`:

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
async for chunk in model.astream([HumanMessage("Hello")]):
    print(chunk)
```

??? note "Streaming Output Options"
    You can append token usage (`prompt_tokens` and `completion_tokens`) at the end of the streaming response via `stream_options={"include_usage": True}`.
    This library enables this option by default; if you need to disable it, you can pass the compatibility option `include_usage=False` when creating the model class or instantiating it.

#### Tool Calling

If the model itself supports tool calling, you can use `bind_tools` directly for tool calling:

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
??? note "Parallel Tool Calling"
    If the model supports parallel tool calling, you can pass `parallel_tool_calls=True` in `bind_tools` to enable parallel tool calling (some model providers enable it by default, so explicit passing is not required).

    For example:

    ```python
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool


    @tool
    def get_current_weather(location: str) -> str:
        """Get the current weather"""
        return f"Current weather in {location} is sunny"
    
    model = ChatVLLM(model="qwen3-4b").bind_tools(
        [get_current_weather], parallel_tool_calls=True
    )
    response = model.invoke([HumanMessage("Get the current weather in Los Angeles and London")])
    print(response)
    ```

??? note "Forcing Tool Calling"

    Through the `tool_choice` parameter, you can control whether the model calls tools and which tool it calls during response, to improve accuracy, reliability, and controllability. Common values include:

    - `"auto"`: The model decides autonomously whether to call tools (default behavior);
    - `"none"`: Prohibit calling tools;
    - `"required"`: Force calling at least one tool;
    - Specify a specific tool (specifically `{"type": "function", "function": {"name": "xxx"}}` in OpenAI compatible APIs).

    Different providers have different ranges of support for `tool_choice`. To resolve differences, this library introduces the compatibility configuration item `supported_tool_choice`, with a default value of `["auto"]`. At this point, `tool_choice` passed in `bind_tools` can only be `auto`, and other values will be filtered out.

    To support passing other `tool_choice` values, the supported items must be configured. The configuration value is a list of strings, with optional values for each string:

    - `"auto"`, `"none"`, `"required"`: Corresponding standard strategies;
    - `"specific"`: Unique identifier for this library, indicating support for specifying a specific tool.

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

    model = ChatVLLM(model="qwen3-4b").bind_tools(
        [get_current_weather], tool_choice="required"
    )
    ```


#### Structured Output

```python
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

model = ChatVLLM(model="qwen3-4b").with_structured_output(User)
response = model.invoke([HumanMessage("Hello, I am Tom, and I am 25 years old")])
print(response)
```
??? note "Default Structured Output Method"

    Currently, there are three common structured output methods: `json_schema`, `function_calling`, and `json_mode`. Among them, `json_schema` has the best effect, so this library's `with_structured_output` prioritizes `json_schema` as the structured output method; it only automatically downgrades to `function_calling` when the provider does not support it. Different model providers have varying degrees of support for structured output. This library uses the compatibility configuration item `supported_response_format` to declare the structured output methods supported by the provider. The default value is `[]`, indicating that neither `json_schema` nor `json_mode` is supported. At this point, `with_structured_output(method=...)` will fix the use of `function_calling`; even if `json_schema` / `json_mode` is passed in, it will automatically be converted to `function_calling`. If you want to use the corresponding structured output method, you need to explicitly pass the corresponding parameters (especially `json_schema`).

    For example, models deployed by vLLM support the `json_schema` structured output method, which can be declared during registration:

    ```python
    from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

    ChatVLLM = create_openai_compatible_model(
        model_provider="vllm",
        chat_model_cls_name="ChatVLLM",
        compatibility_options={"supported_response_format": ["json_schema"]},
    )

    model = ChatVLLM(model="qwen3-4b")
    ``` 

    !!! warning "Attention"
        `supported_response_format` currently only affects the `model.with_structured_output` method. For structured output in `create_agent`, if you need to use the `json_schema` implementation, you need to ensure that the corresponding model's `profile` contains the `structured_output` field with a value of `True`.


#### Passing Extra Parameters

Since this class inherits from `BaseChatOpenAI`, it supports passing model parameters of `BaseChatOpenAI`, such as `temperature`, `extra_body`, etc.

For example, use `extra_body` to pass extra parameters (here to disable thinking mode):

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(
    model="qwen3-4b",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
response = model.invoke([HumanMessage("Hello")])
print(response)
```

#### Passing Multimodal Data

Passing multimodal data is supported. You can use the OpenAI compatible multimodal data format or directly use `content_block` in LangChain.

Passing image type data:

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
Passing video type data:

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
   
#### Using Reasoning Models

A key feature of the model classes created by this library is their enhanced compatibility with a wider range of reasoning models.

For example:

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
response = model.invoke("Why is the feathers of a parrot so vibrant?")
reasoning_steps = [b for b in response.content_blocks if b["type"] == "reasoning"]
print(" ".join(step["reasoning"] for step in reasoning_steps))
```

??? note "Support for Different Reasoning Modes"
    Different models may adopt different reasoning modes: some models require explicit passing of the `reasoning_content` field in the current call, while others do not. This library introduces the `reasoning_keep_policy` compatibility configuration to adapt to these differences.

    This configuration item supports the following values:

    - `never`: **Do not retain any** reasoning content in historical messages (default);

    - `current`: Retain **only the current conversation's** `reasoning_content` field;

    - `all`: Retain **all conversations'** `reasoning_content` field.

    ```mermaid
    graph LR
        A[reasoning_content Retention Policy] --> B{Value?};
        B -->|never| C[Do not include any<br>reasoning_content];
        B -->|current| D[Include only current conversation's<br>reasoning_content<br>Adapt to interleaved thinking mode];
        B -->|all| E[Include all conversations'<br>reasoning_content];
        C --> F[Send to model];
        D --> F;
        E --> F;
    ```

    For example, the user first asks "What is the weather in New York?", then follows up with "What is the weather in London?". Currently, the second round of dialogue is about to take place, and the last model call is imminent.

    - When value is `never`

    The messages passed to the model will **not have any** `reasoning_content` field. The messages received by the model are:

    ```python
    messages = [
        {"content": "What's the weather in New York?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Partly cloudy, 7–13°C", "role": "tool", "tool_call_id": "..."},
        {"content": "It's partly cloudy in New York today, 7–13°C.", "role": "assistant"},
        {"content": "What's the weather in London?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Rainy, 14–20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    - When value is `current`

    Only retain the `reasoning_content` field of the **current conversation**. This policy is suitable for Interleaved Thinking scenarios, where the model alternates between explicit reasoning and tool calls. In this case, the reasoning content of the current round needs to be retained. The messages received by the model are:
    ```python
    messages = [
        {"content": "What's the weather in New York?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "Partly cloudy, 7–13°C", "role": "tool", "tool_call_id": "..."},
        {"content": "It's partly cloudy in New York today, 7–13°C.", "role": "assistant"},
        {"content": "What's the weather in London?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "To check London's weather, I should call the weather tool directly.",  # Only retain current round's reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Rainy, 14–20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    - When value is `all`

    Retain the `reasoning_content` field of **all** conversations. The messages received by the model are:
    ```python
    messages = [
        {"content": "What's the weather in New York?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "To check New York's weather, I should call the weather tool directly.",  # Retain reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Partly cloudy, 7–13°C", "role": "tool", "tool_call_id": "..."},
        {
            "content": "It's partly cloudy in New York today, 7–13°C.",
            "reasoning_content": "Return the weather result for New York directly.",  # Retain reasoning_content
            "role": "assistant",
        },
        {"content": "What's the weather in London?", "role": "user"},
        {
            "content": "",
            "reasoning_content": "To check London's weather, I should call the weather tool directly.",  # Retain reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "Rainy, 14–20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    **Note**: If the current round of conversation does not involve tool calls, `current` and `never` have the same effect.

    It is worth noting that although this parameter belongs to the compatibility configuration item, different models of the same provider, or even the same model in different scenarios, may have different requirements for the `reasoning_content` retention policy. Therefore, **it is recommended to specify explicitly during instantiation**, and there is no need to assign a value when creating the class.

    For example, with the GLM-4.7-Flash model: since it supports Interleaved Thinking, you typically need to set `reasoning_keep_policy` to `current` at instantiation time so that only the current turn's `reasoning_content` is retained. For example:

    ```python
    from langchain_core.messages import HumanMessage

    model = ChatVLLM(model="glm-4.7-flash", reasoning_keep_policy="current")
    bind_model = model.bind_tools(tools=[get_current_weather])
    response = bind_model.invoke([HumanMessage(content="What is the weather in New York?")])
    print(response)
    ```

    GLM-4.7-Flash also supports another reasoning mode called Preserved Thinking. In that case, you need to retain all `reasoning_content` fields from the conversation history. You can set `reasoning_keep_policy` to `all`. For example:

    ```python
    from langchain_core.messages import HumanMessage

    model = ChatVLLM(
        model="glm-4.7-flash",
        reasoning_keep_policy="all",
        extra_body={"chat_template_kwargs": {"clear_thinking": False}},
    )
    bind_model = model.bind_tools(tools=[get_current_weather])
    response = bind_model.invoke([HumanMessage(content="What is the weather in New York?")])
    print(response)
    ```


#### Model Profiles

You can get the model's profile via `model.profile`. By default, it returns an empty dictionary.

You can also explicitly pass the `profile` parameter during instantiation to specify the model profile.

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
Or directly pass the `profile` parameter for all models of the model provider when creating.

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

#### Support for OpenAI's Latest Responses API

This model class also supports OpenAI's latest `responses` API (parameter name `use_responses_api`). Currently, only a few providers support this style of interface; if your provider supports it, you can enable it via `use_responses_api=True`.

For example, if vLLM supports the `responses` API, you can use it like this:

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b", use_responses_api=True)
response = model.invoke([HumanMessage(content="Hello")])
print(response)
```

!!! warning "Attention"
    This feature is not yet guaranteed to be fully supported. It can be used for simple testing but do not use it in a production environment.


!!! warning "Attention"
    This library cannot currently guarantee 100% compatibility with all OpenAI compatible interfaces (although compatibility configurations can be used to improve compatibility). If the model provider already has an official or community integration class, please prioritize that integration class. If you encounter any compatibility issues, feel free to submit an issue in this library's GitHub repository.


## Creation and Usage of Embedding Models

### Creating an Embedding Model Class

Similar to the chat model class, you can use `create_openai_compatible_embedding` to create an embedding model integration class. This function accepts the following parameters:

| Parameter | Description |
|-----------|-------------|
| `embedding_provider` | Embedding model provider name, e.g., `vllm`. Must start with a letter or number, can only contain letters, numbers, and underscores, and must be no more than 20 characters long.<br><br>**Type**: `str`<br>**Required**: Yes |
| `base_url` | Default API address for the model provider.<br><br>**Type**: `str`<br>**Required**: No |
| `embedding_model_cls_name` | Embedding model class name (must conform to Python class naming conventions). Default is `{Provider}Embeddings` (where `{Provider}` is the provider name capitalized).<br><br>**Type**: `str`<br>**Required**: No |

Similarly, we use `create_openai_compatible_embedding` to integrate vLLM's embedding model.

```python
from langchain_dev_utils.embeddings.adapters import create_openai_compatible_embedding

VLLMEmbeddings = create_openai_compatible_embedding(
    embedding_provider="vllm",
    base_url="http://localhost:8000/v1",
    embedding_model_cls_name="VLLMEmbeddings",
)

embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_query("Hello"))
```

`base_url` can also be omitted. If not passed, the library will default to reading the environment variable `VLLM_API_BASE`:

```bash
export VLLM_API_BASE="http://localhost:8000/v1"
```

In this case, the code can omit `base_url`:

```python
from langchain_dev_utils.embeddings.adapters import create_openai_compatible_embedding

VLLMEmbeddings = create_openai_compatible_embedding(
    embedding_provider="vllm",
    embedding_model_cls_name="VLLMEmbeddings",
)

embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_query("Hello"))
```

**Note**: The prerequisite for the above code to run successfully is that the environment variable `VLLM_API_KEY` has been configured. Although vLLM itself does not require an API Key, the embedding model class requires one during initialization. Therefore, please set this variable first, for example:

```bash
export VLLM_API_KEY=vllm_api_key
```

### Using the Embedding Model Class

Here, use the previously created `VLLMEmbeddings` class to initialize an embedding model instance.

#### Vectorizing Query

```python
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_query("Hello"))
```

Similarly, it supports asynchronous calls:

```python
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
res = await embedding.aembed_query("Hello")
print(res)
```

#### Vectorizing String List

```python
documents = ["Hello", "Hello, I am Tom"]
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_documents(documents))
```
Similarly, it supports asynchronous calls:

```python
documents = ["Hello", "Hello, I am Tom"]
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
res = await embedding.aembed_documents(documents)
print(res)
```

!!! warning "Embedding Model Compatibility Note"
    Embedding APIs compatible with OpenAI generally exhibit good compatibility, but you should still pay attention to the following differences:

    1. `check_embedding_ctx_length`: Set to `True` only when using the official OpenAI embedding service; for all other embedding models, set to `False`.

    2. `dimensions`: If the model supports custom dimensions (e.g., 1024, 4096), you can pass this parameter directly.

    3. `chunk_size`: The maximum number of texts that can be processed in a single API call. For example, if `chunk_size` is 10, you can send up to 10 texts in one request for embedding.
    
    4. Single text token limit: Cannot be controlled via parameters; you need to ensure it yourself during the pre-processing chunking stage.

**Note**: The chat model class and embedding model class created using this feature both support passing parameters from `BaseChatOpenAI` and `OpenAIEmbeddings`, such as `temperature`, `extra_body`, `dimensions`, etc.


!!! warning "Attention"
    Similar to model management, the two functions mentioned above use `pydantic.create_model` underneath to create model classes, which incurs some performance overhead. Additionally, `create_openai_compatible_model` uses a global dictionary to record the `profiles` of each model provider. To avoid multi-threading concurrency issues, it is recommended to create the integration classes during the project startup phase and avoid dynamic creation later.


## Integration with Model Management Feature

This library has seamlessly integrated this feature into the model management functionality. When registering a chat model, just set `chat_model` to `"openai-compatible"`; when registering an embedding model, set `embeddings_model` to `"openai-compatible"`.

### Chat Model Class Registration

Specific code is as follows:

**Method 1: Explicit Parameters**

```python
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1"
)
```

**Method 2: Via Environment Variables (Recommended for Configuration Management)**

```python
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible"
    # Automatically reads VLLM_API_BASE
)
```

At the same time, the parameters `base_url`, `compatibility_options`, and `model_profiles` in the `create_openai_compatible_model` function also support being passed in. You just need to pass the corresponding parameters in the `register_model_provider` function.

For example:

```python
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1",
    compatibility_options={
        "supported_tool_choice": ["auto", "none", "required", "specific"],
        "supported_response_format": ["json_schema"]
    },
    model_profiles=model_profiles,
)
```

### Embedding Model Class Registration

Similar to chat model class registration:

**Method 1: Explicit Parameters**

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


!!! success "Best Practice"
    When integrating an OpenAI compatible API, you can directly use `ChatOpenAI` or `OpenAIEmbeddings` from `langchain-openai` and point the `base_url` and `api_key` to your provider's service. This method is simple enough and suitable for relatively simple scenarios (especially when using ordinary chat models rather than reasoning models).

    However, the following issues exist:

    1. Cannot display the chain of thought of non-OpenAI official reasoning models (i.e., content returned by `reasoning_content`).

    2. Does not support `video` type content_block.

    3. The default strategy coverage for structured output is relatively low.

    When you encounter the above differences, you can use the OpenAI compatible integration class provided by this library for adaptation. For embedding models, compatibility is generally better: in most cases, using `OpenAIEmbeddings` directly and setting `check_embedding_ctx_length=False` is sufficient.
