# 对话模型的创建与使用

## 创建对话模型类

使用 `create_openai_compatible_model` 函数可以创建对话模型集成类。该函数接受以下参数：

| 参数 | 说明 |
|------|------|
| `model_provider` | 模型提供商名称，例如 `vllm`。必须以字母或数字开头，只能包含字母、数字和下划线，长度不超过 20 个字符。<br><br>**类型**: `str`<br>**必填**: 是 |
| `base_url` | 模型提供商默认 API 地址。<br><br>**类型**: `str`<br>**必填**: 否 |
| `compatibility_options` | 兼容性选项配置。<br><br>**类型**: `dict`<br>**必填**: 否 |
| `model_profiles` | 该提供商各模型的 profile 配置字典。<br><br>**类型**: `dict`<br>**必填**: 否 |
| `chat_model_cls_name` | 对话模型类名（需符合 Python 类名规范）。默认值为 `Chat{model_provider}`（其中 `{model_provider}` 首字母大写）。<br><br>**类型**: `str`<br>**必填**: 否 |

其中，`compatibility_options` 是一个字典，用于声明该提供商对 OpenAI API 的部分特性的支持情况，以提高兼容性和稳定性。

目前支持以下配置项：

| 配置项 | 说明 |
|--------|------|
| `supported_tool_choice` | 支持的 `tool_choice` 策略列表。<br><br>**类型**: `list[str]`<br>**默认值**: `["auto"]` |
| `supported_response_format` | 支持的 `response_format` 格式列表（`json_schema`、`json_object`）。<br><br>**类型**: `list[str]`<br>**默认值**: `[]` |
| `reasoning_keep_policy` | 历史消息中 `reasoning_content` 字段的保留策略。<br><br>**类型**: `str`<br>**默认值**: `"never"` |
| `reasoning_field_name` | 提供商返回推理内容的字段名，一般无需配置。可选值为 `reasoning_content`或 `reasoning`。<br><br>**类型**: `str`<br>**默认值**: `"reasoning_content"` |
| `include_usage` | 是否在流式返回结果中包含 `usage` 信息。<br><br>**类型**: `bool`<br>**默认值**: `True` |

!!! info "补充"
    由于同一模型提供商的不同模型对 `tool_choice`、`response_format` 等参数的支持情况存在差异，本库将 `supported_tool_choice`、`supported_response_format`、`reasoning_keep_policy` 作为类的**实例属性**。创建对话模型类时可传入默认值，作为该提供商的通用配置；如需针对特定模型微调，可在实例化时覆盖同名参数。
    
    `reasoning_field_name` 与 `include_usage` 为类的私有属性，仅能在创建或注册模型类时通过 `compatibility_options` 传入。


!!! tip "提示"
    本库会基于用户传入的参数，使用内置的 `BaseChatOpenAICompatible` 构建面向特定提供商的对话模型类。该类继承自 `langchain-openai` 的 `BaseChatOpenAI`，并在以下方面进行了增强：

    **1. 支持额外的推理字段 (reasoning_content / reasoning)**
    ChatOpenAI 遵循官方 OpenAI 响应模式，因此无法提取或保留特定于提供商的字段（例如 `reasoning_content`、`reasoning`）。
    本类默认提取并保留 `reasoning_content`，并可通过 `compatibility_options` 中的 `reasoning_field_name` 配置以提取 `reasoning`。

    **2. 动态适配以选择最合适的结构化输出方法**
    OpenAICompatibleChatModel 会根据模型提供商的实际能力，利用 `compatibility_options` 中的 `supported_response_format`，选择最佳的结构化输出方法（`function_calling` 或 `json_schema`）。

    **3. 支持相关参数配置**
    针对参数与官方 OpenAI API 存在差异的情况，本库提供了 `compatibility_options` 参数来解决此问题。
    例如，当不同模型提供商对 `tool_choice` 的支持不一致时，可以通过在 `compatibility_options` 中设置 `supported_tool_choice` 来进行适配。

    **4.支持`video`类型的 content_block**
    补齐 `ChatOpenAI` 在视频类型的 `content_block` 上的能力缺口。



使用如下代码创建对话模型类：

```python hl_lines="4 5 6"
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    base_url="http://localhost:8000/v1",
    chat_model_cls_name="ChatVLLM"
)

model = ChatVLLM(model="qwen2.5-7b")
print(model.invoke("你好"))
```

在创建对话模型类时，`base_url` 参数可以省略。未传入时，本库会默认读取对应环境变量，例如：

```bash
export VLLM_API_BASE=http://localhost:8000/v1
```

此时代码可以省略 `base_url`：

```python hl_lines="4 5"
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    chat_model_cls_name="ChatVLLM",
)

model = ChatVLLM(model="qwen2.5-7b")
print(model.invoke("你好"))
```

**注意**：上述代码成功运行的前提是已配置环境变量 `VLLM_API_KEY`。虽然 vLLM 本身不要求 API Key，但对话模型类初始化时需要传入，因此请先设置该变量，例如：

```bash
export VLLM_API_KEY=vllm_api_key
```

!!! info "提示"
    创建的对话模型类（嵌入模型类也遵循此命名规则）环境变量的命名规则：

    - API 地址：`${PROVIDER_NAME}_API_BASE`（全大写，下划线分隔）。

    - API Key：`${PROVIDER_NAME}_API_KEY`（全大写，下划线分隔）。


## 使用对话模型类

### 普通调用

通过 `invoke` 方法可以进行普通调用，返回模型响应。

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b")
response = model.invoke([HumanMessage("Hello")])
print(response)
```

同时也支持 `ainvoke` 进行异步调用：

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b")
response = await model.ainvoke([HumanMessage("Hello")])
print(response)
```
### 流式调用

通过 `stream` 方法可以进行流式调用，用于流式返回模型响应。

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b")
for chunk in model.stream([HumanMessage("Hello")]):
    print(chunk)
```

以及通过 `astream` 进行异步流式调用：

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b")
async for chunk in model.astream([HumanMessage("Hello")]):
    print(chunk)
```

??? note "流式输出选项"
    可以通过 `stream_options={"include_usage": True}` 在流式响应末尾附加 token 使用情况（`prompt_tokens` 和 `completion_tokens`）。
    本库默认开启该选项；若需关闭，可在创建模型类传入兼容性选项 `include_usage=False`。

### 工具调用

如果模型本身支持工具调用，可以直接使用 `bind_tools` 进行工具调用：

```python
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import datetime

@tool
def get_current_time() -> str:
    """获取当前时间戳"""
    return str(datetime.datetime.now().timestamp())

model = ChatVLLM(model="qwen2.5-7b").bind_tools([get_current_time])
response = model.invoke([HumanMessage("获取当前时间戳")])
print(response)
```
??? note "并行工具调用"
    如果模型支持并行工具调用，可以在 `bind_tools` 中传递 `parallel_tool_calls=True` 开启并行工具调用（部分模型提供商默认开启，则无需显式传参）。
    
    例如：

    ```python hl_lines="11"
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool


    @tool
    def get_current_weather(location: str) -> str:
        """获取当前天气"""
        return f"当前{location}的天气是晴朗"
    
    model = ChatVLLM(model="qwen2.5-7b").bind_tools(
        [get_current_weather], parallel_tool_calls=True
    )
    response = model.invoke([HumanMessage("获取洛杉矶和伦敦的天气")])
    print(response)
    ```

??? note "强制工具调用"

    通过 `tool_choice` 参数，可以控制模型在响应时是否调用工具以及调用哪个工具，以提升准确性、可靠性和可控性。常见取值有：

    - `"auto"`：模型自主决定是否调用工具（默认行为）；
    - `"none"`：禁止调用工具；
    - `"required"`：强制调用至少一个工具；
    - 指定具体工具（在 OpenAI 兼容 API 中，具体为 `{"type": "function", "function": {"name": "xxx"}}`）。

    不同提供商对`tool_choice`的支持范围不同。为解决差异，本库引入兼容性配置项 `supported_tool_choice`，默认值为 `["auto"]`，此时`bind_tools` 中传入的 `tool_choice` 只能为 `auto`，其他取值会被过滤。

    若需支持传递其他 `tool_choice` 取值，必须配置支持项。配置值为字符串列表，每个字符串的可选值：

    - `"auto"`, `"none"`, `"required"`：对应标准策略；
    - `"specific"`：本库特有标识，表示支持指定具体工具。

    例如 vLLM 支持全部策略：

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


### 结构化输出

```python
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

model = ChatVLLM(model="qwen2.5-7b").with_structured_output(User)
response = model.invoke([HumanMessage("你好，我叫张三，今年25岁")])
print(response)

```
??? note "默认的结构化输出方法"

    目前常见的结构化输出方法有三种：`json_schema`、`function_calling`、`json_mode`。其中，效果最好的是`json_schema`，故本库的 `with_structured_output` 会优先使用`json_schema`作为结构化输出方法；当提供商不支持时，才会自动降级为 `function_calling`。不同的模型提供商对于结构化输出的支持程度有所不同。本库通过兼容性配置项 `supported_response_format` 声明提供商支持的结构化输出方法。默认值为 `[]`，表示既不支持 `json_schema` 也不支持 `json_mode`。此时 `with_structured_output(method=...)` 会固定使用 `function_calling`；即使传入 `json_schema` / `json_mode` 也会自动转化为 `function_calling`，如果想要使用对应的结构化输出方法，需要显式传入响应的参数（尤其是`json_schema`)。

    例如，vLLM 部署的模型支持 `json_schema` 结构化输出方法，则可以在注册时进行声明：

    ```python hl_lines="6"
    from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

    ChatVLLM = create_openai_compatible_model(
        model_provider="vllm",
        chat_model_cls_name="ChatVLLM",
        compatibility_options={"supported_response_format": ["json_schema"]},
    )

    model = ChatVLLM(model="qwen2.5-7b")
    ``` 

    !!! note "注意"
        若 `supported_response_format` 包含 `json_schema`，则 `model.profile` 中的 `structured_output` 字段将自动置为 `True`，此时使用 `create_agent` 进行结构化输出时如果未指定具体的结构化输出策略，默认会采用 `json_schema` 作为结构化输出策略。

        例如: 
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

        输出结果为

        ```
        {'structured_output': True}
        ```


### 传递额外参数

由于该类继承自 `BaseChatOpenAI`，因此支持传递 `BaseChatOpenAI` 的模型参数，例如 `temperature`、`extra_body` 等。

对于非OpenAI官方定义的请求参数，可以通过 `extra_body` 传递。

例如：

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(
    model="qwen2.5-7b",
    extra_body={"top_k": 50},
)
response = model.invoke([HumanMessage("Hello")])
print(response)
```

### 传递多模态数据

支持传递多模态数据。你可以使用 OpenAI 兼容的多模态数据格式，或直接使用 LangChain 中的 `content_block`。

传递图片类数据：

```python
from langchain_core.messages import HumanMessage
messages = [
    HumanMessage(
        content_blocks=[
            {
                "type": "image",
                "url": "https://example.com/image.png",
            },
            {"type": "text", "text": "描述这张图片"},
        ]
    )
]

model = ChatVLLM(model="qwen2.5-vl-7b")
response = model.invoke(messages)
print(response)
```
传递视频类数据：

```python
from langchain_core.messages import HumanMessage

messages = [
    HumanMessage(
        content_blocks=[
            {
                "type": "video",
                "url": "https://example.com/video.mp4",
            },
            {"type": "text", "text": "描述这段视频"},
        ]
    )
]

model = ChatVLLM(model="qwen2.5-vl-7b")
response = model.invoke(messages)
print(response)
```
   
### 使用推理模型

本库创建的模型类的一大特点就是进一步适配了更多的推理模型。例如接入`qwen3-4b`模型。

例如：

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
response = model.invoke("为什么鹦鹉的羽毛如此鲜艳？")
reasoning_steps = [b for b in response.content_blocks if b["type"] == "reasoning"]
print(" ".join(step["reasoning"] for step in reasoning_steps))
```
!!! note "注意"
    由于新版的vLLM默认将推理内容以`reasoning`字段返回，因此如果使用vLLM部署的推理模型，必须在创建对话模型类时指定`reasoning_field_name`参数，取值为`reasoning`。

    但为复用既有的 `content_blocks` 默认解析逻辑，本库仍然会将其保存至 `additional_kwargs["reasoning_content"]`。


??? note "不同推理模式的支持"

    不同模型的推理模式不尽相同（这点在Agent开发中尤为重要）：有些需要在本次调用中显式传递推理内容，有些则无需。本库提供 `reasoning_keep_policy` 兼容性配置以适配这些差异。

    该配置项支持以下取值：

    - `never`：在历史消息中**不保留任何**推理内容（默认)；

    - `current`：仅保留**当前对话**中的推理内容；

    - `all`：保留**所有对话**中的推理内容。

    ```mermaid
    graph LR
        A[reasoning_content 保留策略] --> B{取值?};
        B -->|never| C[不包含任何<br>推理内容];
        B -->|current| D[仅包含当前对话的<br>推理内容<br>适配交错式思考模式];
        B -->|all| E[包含所有对话的<br>推理内容];
        C --> F[发送给模型];
        D --> F;
        E --> F;
    ```
    
    例如，假设推理内容的字段名称是 `reasoning_content`，当用户先提问"纽约天气如何？"，随后追问"伦敦天气如何？"，当前正要进行第二轮对话，且即将进行最后一次模型调用。

    - 取值为`never`时

    最终传递给模型的 messages 中**不会有任何** `reasoning_content` 字段，模型收到的 messages 为：

    ```python
    messages = [
        {"content": "查纽约天气如何？", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "多云 7~13°C", "role": "tool", "tool_call_id": "..."},
        {"content": "纽约今天天气为多云，7~13°C。", "role": "assistant"},
        {"content": "查伦敦天气如何？", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "雨天，14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    - 取值为`current`时

    仅保留**当前对话**中的 `reasoning_content` 字段。该策略适用于交错式思考（Interleaved Thinking）场景，即模型在显式推理与工具调用之间交替进行，此时需要将当前轮次的推理内容进行保留。模型收到的 messages 为：
    ```python
    messages = [
        {"content": "查纽约天气如何？", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [...]},
        {"content": "多云 7~13°C", "role": "tool", "tool_call_id": "..."},
        {"content": "纽约今天天气为多云，7~13°C。", "role": "assistant"},
        {"content": "查伦敦天气如何？", "role": "user"},
        {
            "content": "",
            "reasoning_content": "查伦敦天气，需要直接调用天气工具。",  # 仅保留本轮对话的 reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "雨天，14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    - 取值为`all`时

    保留**所有**对话中的 `reasoning_content` 字段。模型收到的 messages 为：
    ```python
    messages = [
        {"content": "查纽约天气如何？", "role": "user"},
        {
            "content": "",
            "reasoning_content": "查纽约天气，需要直接调用天气工具。",  # 保留 reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "多云 7~13°C", "role": "tool", "tool_call_id": "..."},
        {
            "content": "纽约今天天气为多云，7~13°C。",
            "reasoning_content": "直接返回纽约天气结果。",  # 保留 reasoning_content
            "role": "assistant",
        },
        {"content": "查伦敦天气如何？", "role": "user"},
        {
            "content": "",
            "reasoning_content": "查伦敦天气，需要直接调用天气工具。",  # 保留 reasoning_content
            "role": "assistant",
            "tool_calls": [...],
        },
        {"content": "雨天，14~20°C", "role": "tool", "tool_call_id": "..."},
    ]
    ```

    **注意**：若本轮对话不涉及工具调用，则`current`与`never`效果相同。

    值得注意的是，虽然该参数属于兼容性配置项，但同一提供商的不同模型、甚至同一模型在不同场景下对推理内容的保留策略要求也可能不同，因此**建议在实例化时显式指定**，创建类时无需赋值。

    例如，以 GLM-4.7-Flash 模型为例，由于其支持交错式思考（Interleaved Thinking）模式，一般需要在实例化时将 `reasoning_keep_policy` 设置为 `current`，以便仅保留当前轮次的推理内容。例如：

    ```python hl_lines="3"
    from langchain_core.messages import HumanMessage

    model = ChatVLLM(model="glm-4.7-flash", reasoning_keep_policy="current")
    agent = create_agent(
        model=model,
        tools=[get_current_weather],
    )
    response = agent.invoke({"messages": [HumanMessage(content="查纽约天气如何？")]})
    print(response)
    ```
    同时，GLM-4.7-Flash 模型也支持另一种思考模式，被称之为Preserved Thinking。此时需要保留历史消息中的所有 `reasoning_content` 字段，可以将 `reasoning_keep_policy` 设置为 `all`。例如：

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
    response = agent.invoke({"messages": [HumanMessage(content="查纽约天气如何？")]})
    print(response)
    ```

    !!! note "注意"
       同样，GLM-4.7-Flash 模型也为推理模型，且回传推理内容的字段为 `reasoning`。故同样需要在创建对话模型类时指定兼容性选项`reasoning_field_name`取值为`reasoning`。


### Model profiles

可以通过 `model.profile` 获取模型的 profile。默认情况下返回空字典。

你也可以在实例化时显式传入 `profile` 参数指定模型 profile。

例如：
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
或者直接在创建时传入模型提供商的所有模型的`profile`参数。

例如：
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
    # 此处还可以写更多的model profile
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

### 支持 OpenAI 最新的 Responses API

该模型类也支持 OpenAI 最新的 `responses` API（参数名为 `use_responses_api`）。目前仅少量提供商支持该风格接口；若你的提供商支持，可通过 `use_responses_api=True` 开启。

例如 vLLM 支持 `responses` API，则可以这样使用：

```python hl_lines="3"
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen2.5-7b", use_responses_api=True)
response = model.invoke([HumanMessage(content="你好")])
print(response)
```

目前该功能的实现完全依托于`BaseChatOpenAI`对于`responses` API的实现，故在使用中可能存在一定的兼容性问题，后续将会根据实际情况进行优化。


!!! warning "注意"
    本库尚无法保证对所有 OpenAI 兼容接口的 100% 兼容（虽然可通过兼容性配置尽量提升，但仍可能存在差异）。若目标模型已有官方或社区维护的集成类，请优先使用。如遇兼容性问题，欢迎在 GitHub 仓库提 issue。

    以 OpenRouter 为例，其虽提供 OpenAI 兼容接口，但存在多项兼容差异；LangChain 官方已推出 [ChatOpenRouter](https://docs.langchain.com/oss/python/integrations/providers/openrouter)，建议直接采用该类接入 OpenRouter。

!!! warning "注意"
    该函数底层使用 `pydantic.create_model` 创建对话模型类，会带来一定的性能开销。此外，`create_openai_compatible_model` 使用全局字典记录各模型提供商的 `profiles`，为了避免多线程并发问题，建议在项目启动阶段创建好集成类，后续避免动态创建。

!!! success "最佳实践"
    接入 OpenAI 兼容 API 的对话模型提供商时，可以直接使用 `langchain-openai` 的 `ChatOpenAI`，并通过 `base_url` 与 `api_key` 指向你的提供商服务。该方式足够简单，适用于比较简单的场景（尤其是使用的是普通的对话模型而不是推理模型）。

    但是会存在以下问题：

    1. 无法显示非 OpenAI 官方推理模型的思维链（即`reasoning_content` / `reasoning` 返回的内容）

    2. 不支持 `video` 类型的 content_block

    3. 结构化输出的默认策略覆盖率较低

    当你遇到上述差异时，可以使用本库提供的 OpenAI 兼容集成类进行适配。
