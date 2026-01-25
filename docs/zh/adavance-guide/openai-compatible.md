# OpenAI 兼容 API 模型提供商集成

!!! warning "前提条件"
    使用此功能时，必须安装 standard 版本的 `langchain-dev-utils` 库。具体可以参考安装部分的介绍。

## 概述

许多模型提供商都提供 **OpenAI 兼容 API** 服务，例如 [vLLM](https://github.com/vllm-project/vllm)、[OpenRouter](https://openrouter.ai/) 和 [Together AI](https://www.together.ai/) 等。本库提供一套 OpenAI 兼容 API 集成方案，覆盖对话模型与嵌入模型，尤其适用于「提供商已提供 OpenAI 兼容 API，但尚无对应 LangChain 集成」的场景。

本库提供了两个工具函数，用于创建对话模型集成类与嵌入模型集成类：

| 函数名 | 说明 |
|--------|------|
| `create_openai_compatible_model` | 创建对话模型集成类 |
| `create_openai_compatible_embedding` | 创建嵌入模型集成类 |


!!! tip "说明"
    本库提供的两个工具函数的最初灵感借鉴自 JavaScript 生态的 [@ai-sdk/openai-compatible](https://ai-sdk.dev/providers/openai-compatible-providers)。

以下将以接入 [vLLM](https://github.com/vllm-project/vllm) 为例，展示如何使用本功能。

??? note "vLLM 介绍"
    vLLM 是常用的大模型推理框架，适合本地或自建环境下的高性能推理服务。它可以将大模型部署为 OpenAI 兼容的 API，便于复用现有的 SDK 与调用方式；同时支持对话模型与嵌入模型的部署，以及多模型服务、工具调用与推理输出等能力，适用于对话、工具调用与多模态等场景。

    以下示例均为后续内容中会用到的模型部署命令：
    
    **Qwen3-4B**：

    ```bash
    vllm serve Qwen/Qwen3-4B \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3-4b
    ```

    **GLM-4.7-Flash**：

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

    **Qwen3-VL-2B-Instruct**：

    ```bash
    vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3-vl-2b
    ```

    **Qwen3-Embedding-4B**：

    ```bash
    vllm serve Qwen/Qwen3-Embedding-4B \
    --task embed \
    --served-model-name qwen3-embedding-4b \
    --host 0.0.0.0 --port 8000
    ```
    服务地址为 `http://localhost:8000/v1`。


## 对话模型的创建与使用

### 创建对话模型类

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
| `include_usage` | 是否在流式返回结果中包含 `usage` 信息。<br><br>**类型**: `bool`<br>**默认值**: `True` |

!!! info "补充"
    由于同一模型提供商的不同模型对 `tool_choice`、`response_format` 等参数的支持情况存在差异，这四个兼容性选项为类的**实例属性**。因此，创建对话模型类时可以传入值作为全局默认值（代表该提供商大部分模型支持的配置），后续如需针对特定模型进行微调，可在实例化时覆盖同名参数。


!!! tip "提示"
    本库会基于用户传入的参数，使用内置的 `BaseChatOpenAICompatible` 构建面向特定提供商的对话模型类。该类继承自 `langchain-openai` 的 `BaseChatOpenAI`，并在以下方面进行了增强：

    - **支持更多格式的推理内容**：除 OpenAI 官方格式外，也支持以 `reasoning_content` 参数返回的推理内容格式。
    - **支持 `video` 类型的 content_block**：补齐 `ChatOpenAI` 在视频类型的 `content_block` 上的能力缺口。
    - **自动选择更合适的结构化输出方式**：根据提供商实际支持情况，在 `function_calling` 与 `json_schema` 之间自动选择更优方案。
    - **通过 `compatibility_options` 精细适配差异**：按需配置对 `tool_choice`、`response_format` 等参数的支持差异。


使用如下代码创建对话模型类：

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

在创建对话模型类时，`base_url` 参数可以省略。未传入时，本库会默认读取对应环境变量，例如：

```bash
export VLLM_API_BASE=http://localhost:8000/v1
```

此时代码可以省略 `base_url`：

```python
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    chat_model_cls_name="ChatVLLM",
)

model = ChatVLLM(model="qwen3-4b")
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


### 使用对话模型类

#### 普通调用

通过 `invoke` 方法可以进行普通调用，返回模型响应。

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
response = model.invoke([HumanMessage("Hello")])
print(response)
```

同时也支持 `ainvoke` 进行异步调用：

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
response = await model.ainvoke([HumanMessage("Hello")])
print(response)
```
#### 流式调用

通过 `stream` 方法可以进行流式调用，用于流式返回模型响应。

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
for chunk in model.stream([HumanMessage("Hello")]):
    print(chunk)
```

以及通过 `astream` 进行异步流式调用：

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
async for chunk in model.astream([HumanMessage("Hello")]):
    print(chunk)
```

??? note "流式输出选项"
    可以通过 `stream_options={"include_usage": True}` 在流式响应末尾附加 token 使用情况（`prompt_tokens` 和 `completion_tokens`）。
    本库默认开启该选项；若需关闭，可在创建模型类或实例化时传入兼容性选项 `include_usage=False`。

#### 工具调用

如果模型本身支持工具调用，可以直接使用 `bind_tools` 进行工具调用：

```python
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import datetime

@tool
def get_current_time() -> str:
    """获取当前时间戳"""
    return str(datetime.datetime.now().timestamp())

model = ChatVLLM(model="qwen3-4b").bind_tools([get_current_time])
response = model.invoke([HumanMessage("获取当前时间戳")])
print(response)
```
??? note "并行工具调用"
    如果模型支持并行工具调用，可以在 `bind_tools` 中传递 `parallel_tool_calls=True` 开启并行工具调用（部分模型提供商默认开启，则无需显式传参）。
    
    例如：

    ```python
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool


    @tool
    def get_current_weather(location: str) -> str:
        """获取当前天气"""
        return f"当前{location}的天气是晴朗"
    
    model = ChatVLLM(model="qwen3-4b").bind_tools(
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


#### 结构化输出

```python
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

model = ChatVLLM(model="qwen3-4b").with_structured_output(User)
response = model.invoke([HumanMessage("你好，我叫张三，今年25岁")])
print(response)
```
??? note "默认的结构化输出方法"

    目前常见的结构化输出方法有三种：`json_schema`、`function_calling`、`json_mode`。其中，效果最好的是`json_schema`，故本库的 `with_structured_output` 会优先使用`json_schema`作为结构化输出方法；当提供商不支持时，才会自动降级为 `function_calling`。不同的模型提供商对于结构化输出的支持程度有所不同。本库通过兼容性配置项 `supported_response_format` 声明提供商支持的结构化输出方法。默认值为 `[]`，表示既不支持 `json_schema` 也不支持 `json_mode`。此时 `with_structured_output(method=...)` 会固定使用 `function_calling`；即使传入 `json_schema` / `json_mode` 也会自动转化为 `function_calling`，如果想要使用对应的结构化输出方法，需要显式传入响应的参数（尤其是`json_schema`)。

    例如，vLLM 部署的模型支持 `json_schema` 结构化输出方法，则可以在注册时进行声明：

    ```python
    from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

    ChatVLLM = create_openai_compatible_model(
        model_provider="vllm",
        chat_model_cls_name="ChatVLLM",
        compatibility_options={"supported_response_format": ["json_schema"]},
    )

    model = ChatVLLM(model="qwen3-4b")
    ``` 

    !!! warning "注意"
        `supported_response_format`目前仅影响`model.with_structured_output`方法。对于`create_agent`中的结构化输出，若需要使用`json_schema`的实现方式，你需要确保对应模型的`profile`中包含`structured_output`字段，且值为`True`。


#### 传递额外参数

由于该类继承自 `BaseChatOpenAI`，因此支持传递 `BaseChatOpenAI` 的模型参数，例如 `temperature`、`extra_body` 等。

例如，使用 `extra_body` 传递额外参数（此处为关闭思考模式）：

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(
    model="qwen3-4b",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
response = model.invoke([HumanMessage("Hello")])
print(response)
```

#### 传递多模态数据

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

model = ChatVLLM(model="qwen3-vl-2b")
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

model = ChatVLLM(model="qwen3-vl-2b")
response = model.invoke(messages)
print(response)
```
   
#### 使用推理模型

本库创建的模型类的一大特点就是进一步适配了更多的推理模型。

例如：

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b")
response = model.invoke("为什么鹦鹉的羽毛如此鲜艳？")
reasoning_steps = [b for b in response.content_blocks if b["type"] == "reasoning"]
print(" ".join(step["reasoning"] for step in reasoning_steps))
```

??? note "不同推理模式的支持"

    不同模型的推理模式不尽相同（这点在Agent开发中尤为重要）：有些需要在本次调用中显式传递 `reasoning_content` 字段，有些则无需。本库提供 `reasoning_keep_policy` 兼容性配置以适配这些差异。

    该配置项支持以下取值：

    - `never`：在历史消息中**不保留任何**推理内容（默认)；

    - `current`：仅保留**当前对话**中的 `reasoning_content` 字段；

    - `all`：保留**所有对话**中的 `reasoning_content` 字段。

    ```mermaid
    graph LR
        A[reasoning_content 保留策略] --> B{取值?};
        B -->|never| C[不包含任何<br>reasoning_content];
        B -->|current| D[仅包含当前对话的<br>reasoning_content<br>适配交错式思考模式];
        B -->|all| E[包含所有对话的<br>reasoning_content];
        C --> F[发送给模型];
        D --> F;
        E --> F;
    ```

    例如，用户先提问"纽约天气如何？"，随后追问"伦敦天气如何？"，当前正要进行第二轮对话，且即将进行最后一次模型调用。

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

    值得注意的是，虽然该参数属于兼容性配置项，但同一提供商的不同模型、甚至同一模型在不同场景下对 `reasoning_content` 的保留策略要求也可能不同，因此**建议在实例化时显式指定**，创建类时无需赋值。

    例如，以 GLM-4.7-Flash 模型为例，由于其支持交错式思考（Interleaved Thinking）模式，一般需要在实例化时将 `reasoning_keep_policy` 设置为 `current`，以便仅保留当前轮次的 `reasoning_content`。例如：

    ```python
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

    ```python
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


#### Model profiles

可以通过 `model.profile` 获取模型的 profile。默认情况下返回空字典。

你也可以在实例化时显式传入 `profile` 参数指定模型 profile。

例如：
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
或者直接在创建时传入模型提供商的所有模型的`profile`参数。

例如：
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
    # 此处还可以写更多的model profile
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

#### 支持 OpenAI 最新的 Responses API

该模型类也支持 OpenAI 最新的 `responses` API（参数名为 `use_responses_api`）。目前仅少量提供商支持该风格接口；若你的提供商支持，可通过 `use_responses_api=True` 开启。

例如 vLLM 支持 `responses` API，则可以这样使用：

```python
from langchain_core.messages import HumanMessage

model = ChatVLLM(model="qwen3-4b", use_responses_api=True)
response = model.invoke([HumanMessage(content="你好")])
print(response)
```

!!! warning "注意"
    该功能暂未保证完全支持，可以用于简单测试，但不要用于生产环境。


!!! warning "注意"
    本库目前无法保证 100% 适配所有 OpenAI 兼容接口（尽管可以使用兼容性配置来提升兼容性）。若模型提供商已有官方或社区集成类，请优先采用该集成类。如遇到任何兼容性问题，欢迎在本库 GitHub 仓库提交 issue。


## 嵌入模型的创建与使用

### 创建嵌入模型类

与对话模型类类似，可以使用 `create_openai_compatible_embedding` 创建嵌入模型集成类。该函数接受以下参数：

| 参数 | 说明 |
|------|------|
| `embedding_provider` | 嵌入模型提供商名称，例如 `vllm`。必须以字母或数字开头，只能包含字母、数字和下划线，长度不超过 20 个字符。<br><br>**类型**: `str`<br>**必填**: 是 |
| `base_url` | 模型提供商默认 API 地址。<br><br>**类型**: `str`<br>**必填**: 否 |
| `embedding_model_cls_name` | 嵌入模型类名（需符合 Python 类名规范）。默认值为 `{Provider}Embeddings`（其中 `{Provider}` 为首字母大写的提供商名称）。<br><br>**类型**: `str`<br>**必填**: 否 |

同样，我们使用 `create_openai_compatible_embedding` 来集成 vLLM 的嵌入模型。

```python
from langchain_dev_utils.embeddings.adapters import create_openai_compatible_embedding

VLLMEmbeddings = create_openai_compatible_embedding(
    embedding_provider="vllm",
    base_url="http://localhost:8000/v1",
    embedding_model_cls_name="VLLMEmbeddings",
)

embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_query("你好"))
```

`base_url` 也可以省略。未传入时，本库会默认读取环境变量 `VLLM_API_BASE`：

```bash
export VLLM_API_BASE="http://localhost:8000/v1"
```

此时代码可以省略 `base_url`：

```python
from langchain_dev_utils.embeddings.adapters import create_openai_compatible_embedding

VLLMEmbeddings = create_openai_compatible_embedding(
    embedding_provider="vllm",
    embedding_model_cls_name="VLLMEmbeddings",
)

embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_query("你好"))
```

**注意**：上述代码成功运行的前提是已配置环境变量 `VLLM_API_KEY`。虽然 vLLM 本身不要求 API Key，但嵌入模型类初始化时需要传入，因此请先设置该变量，例如：

```bash
export VLLM_API_KEY=vllm_api_key
```

### 使用嵌入模型类

这里使用前面创建好的 `VLLMEmbeddings` 类来初始化嵌入模型实例。

#### 向量化查询

```python
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_query("你好"))
```

同样，也支持异步调用：

```python
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
res = await embedding.aembed_query("你好")
print(res)
```

#### 向量化字符串列表

```python
documents = ["你好", "你好，我是张三"]
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_documents(documents))
```
同样，也支持异步调用：

```python
documents = ["你好", "你好，我是张三"]
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
res = await embedding.aembed_documents(documents)
print(res)
```

!!! warning "嵌入模型兼容性说明"
    兼容 OpenAI 的嵌入 API 通常表现出较好的兼容性，但仍需注意以下差异点：

    1. `check_embedding_ctx_length`：仅在使用官方 OpenAI 嵌入服务时设为 `True`；其余嵌入模型一律设为 `False`。

    2. `dimensions`：若模型支持自定义向量维度（如 1024、4096），可直接传入该参数。

    3. `chunk_size`：为单次API调用中能处理的文本数量上限。例如`chunk_size`大小为10，意味着一次请求最多可传入10个文本进行向量化。
    
    4. 单文本 token 上限：无法通过参数控制，需在预处理分块阶段自行保证。

**注意**：使用本功能创建的对话模型类和嵌入模型类，均支持传入 `BaseChatOpenAI` 与 `OpenAIEmbeddings` 的参数，例如 `temperature`、`extra_body`、`dimensions` 等。


!!! warning "注意"
    与模型管理类似，上述两个函数底层使用 `pydantic.create_model` 创建模型类，会带来一定的性能开销。此外，`create_openai_compatible_model` 使用全局字典记录各模型提供商的 `profiles`，为了避免多线程并发问题，建议在项目启动阶段创建好集成类，后续避免动态创建。


## 与模型管理功能集成

本库已将此功能无缝接入模型管理功能。注册对话模型时，只需将 `chat_model` 设为 `"openai-compatible"`；注册嵌入模型时，将 `embeddings_model` 设为 `"openai-compatible"` 即可。

### 对话模型类注册

具体代码如下：

**方式一：显式传参**

```python
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1"
)
```

**方式二：通过环境变量（推荐用于配置管理）**

```python
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible"
    # 自动读取 VLLM_API_BASE
)
```

同时，`create_openai_compatible_model`函数中的`base_url`、`compatibility_options`、`model_profiles`参数也支持传入。只需要在`register_model_provider`函数中传入对应的参数即可。

例如：

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

### 嵌入模型类注册

与对话模型类注册类似：

**方式一：显式传参**

```python
from langchain_dev_utils.embeddings import register_embeddings_provider

register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)
```

**方式二：环境变量（推荐）**

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


!!! success "最佳实践"
    接入 OpenAI 兼容 API 时，可以直接使用 `langchain-openai` 的 `ChatOpenAI` 或 `OpenAIEmbeddings`，并通过 `base_url` 与 `api_key` 指向你的提供商服务。该方式足够简单，适用于比较简单的场景（尤其是使用的是普通的对话模型而不是推理模型）。

    但是会存在以下问题：

    1. 无法显示非 OpenAI 官方推理模型的思维链（即`reasoning_content`返回的内容）

    2. 不支持 `video` 类型的 content_block

    3. 结构化输出的默认策略覆盖率较低

    当你遇到上述差异时，可以使用本库提供的 OpenAI 兼容集成类进行适配。对于嵌入模型，兼容性通常更好：多数情况下直接使用 `OpenAIEmbeddings` 并将 `check_embedding_ctx_length=False` 即可。
