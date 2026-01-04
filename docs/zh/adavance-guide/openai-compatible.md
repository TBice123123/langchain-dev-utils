# OpenAI 兼容 API 模型提供商集成

## 概述

很多模型提供商都支持**OpenAI 兼容**的API服务，例如：[vLLM](https://github.com/vllm-project/vllm)、[OpenRouter](https://openrouter.ai/)、[Together AI](https://www.together.ai/)等。本库提供了完整的OpenAI兼容API集成方案，支持对话模型和嵌入模型，特别适用于暂时没有对应的LangChain集成而提供商提供OpenAI兼容API的场景。


!!! tip "提示"
    接入 OpenAI 兼容 API 的常见做法是直接使用 `langchain-openai` 中的 `ChatOpenAI` 或 `OpenAIEmbeddings`，只需传入 `base_url` 与 `api_key` 即可。然而，这种方式仅适用于简单场景，存在诸多兼容性问题，尤其是对话模型，具体包括：

    1. 无法显示非 OpenAI 官方推理模型的思维链（`reasoning_content`）
    2. 不支持视频类型的 content_block
    3. 结构化输出默认策略覆盖率低

    本库提供此功能正是为了解决上述兼容性问题。对于简单场景（尤其是对兼容性要求不高的场景），可直接使用 `ChatOpenAI`，无需使用本功能。`OpenAIEmbeddings` 兼容性较好，只需将 `check_embedding_ctx_length` 设为 `False` 即可。此外，为方便开发者，我们也提供了嵌入模型的 OpenAI 兼容集成类功能。

## 创建对应的集成类

本库提供了两个工具函数，用于创建对应的对话模型集成类和嵌入模型集成类。具体为：

| 函数名 | 说明 |
|--------|------|
| `create_openai_compatible_model` | 创建对话模型集成类 |
| `create_openai_compatible_embedding` | 创建嵌入模型集成类 |


!!! tip "提示"
    本库提供的两个工具函数的最初灵感借鉴自 JavaScript 生态的 [@ai-sdk/openai-compatible](https://ai-sdk.dev/providers/openai-compatible-providers)。

### 创建对话模型类

使用 `create_openai_compatible_model` 函数可以创建对话模型集成类。该函数接受以下参数：

| 参数 | 说明 |
|------|------|
| `model_provider` | 模型提供商名称，例如 `vllm`。<br><br>**类型**: `str`<br>**必填**: 是 |
| `base_url` | 模型提供商的默认API地址。<br><br>**类型**: `str`<br>**必填**: 否 |
| `compatibility_options` | 兼容性选项配置。<br><br>**类型**: `dict`<br>**必填**: 否 |
| `model_profiles` | 该模型提供商所提供的模型对应的profiles。<br><br>**类型**: `dict`<br>**必填**: 否 |
| `chat_model_cls_name` | 对话模型类名(需要符合Python类名规范)，默认值为 `Chat{model_provider}`（其中 `{model_provider}` 首字母大写）。<br><br>**类型**: `str`<br>**必填**: 否 |

!!! warning "注意"
    `model_provider` 必须以字母或数字开头，只能包含字母、数字和下划线，长度不超过 20 个字符。(`embedding_provider` 同理)

本库会根据用户传入的上述参数，使用内置 `BaseChatOpenAICompatible` 类构建对应于特定提供商的对话模型类。该类继承自 `langchain-openai` 的 `BaseChatOpenAI`，并增强以下能力：

- **支持更多格式的推理内容**：相较于 `ChatOpenAI` 只能输出官方的推理内容，本类还支持输出更多格式的推理内容（例如 `vLLM`）。
- **支持 `video` 类型 content_block**：`ChatOpenAI` 无法转换 `type=video` 的 `content_block`，本实现已完成支持。
- **动态适配并选择最合适的结构化输出方法**：默认情况下，能够根据模型提供商的实际支持情况，自动选择最优的结构化输出方法（`function_calling` 或 `json_schema`）。
- **通过 compatibility_options 精细适配差异**：通过配置提供商兼容性选项，解决 `tool_choice`、`response_format` 等参数的支持差异。

!!! warning "注意"
    使用此功能时，必须安装 standard 版本的 `langchain-dev-utils` 库。具体可以参考安装部分的介绍。

#### 代码示例

我们以集成 vLLM 为例，展示如何使用 `create_openai_compatible_model` 函数创建对话模型集成类。

!!! note "补充"
    vLLM 是常用的大模型推理框架，其可以将大模型部署为 OpenAI 兼容的 API，例如本例子中的 Qwen3-4B：

    ```bash
    vllm serve Qwen/Qwen3-4B \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3-4b
    ```
    服务地址为 `http://localhost:8000/v1`。


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

值得注意的是，在创建对话模型类时，`base_url`参数可以省略，若未传入，本库会默认读取对应的环境变量。例如

```bash
export VLLM_API_BASE=http://localhost:8000/v1
```

此时代码可以省略 `base_url` 参数：

```python
from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    chat_model_cls_name="ChatVLLM",
)

model = ChatVLLM(model="qwen3-4b")
print(model.invoke("你好"))
```

**注意**：上述代码成功运行的前提是已配置环境变量 `VLLM_API_KEY`。虽然 vLLM 本身不要求 API Key，但对话模型类初始化时必须传入，因此请先设置该变量。例如

```bash
export VLLM_API_KEY=vllm_api_key
```


!!! info "提示"
    创建的对话模型类（嵌入模型类也遵循此命名规则）环境变量的命名规则：

    - API地址：`${PROVIDER_NAME}_API_BASE`（全大写，下划线分隔）。

    - API Key：`${PROVIDER_NAME}_API_KEY`（全大写，下划线分隔）。


#### 兼容性参数

`compatibility_options` 是一个字典，用于声明该提供商对 OpenAI API 的部分特性的支持情况，以提高兼容性和稳定性。

目前支持以下配置项：

| 配置项 | 说明 |
|--------|------|
| `supported_tool_choice` | 支持的 `tool_choice` 策略列表。<br><br>**类型**: `list[str]`<br>**默认值**: `["auto"]` |
| `supported_response_format` | 支持的 `response_format` 格式列表(`json_schema`、`json_object`)。<br><br>**类型**: `list[str]`<br>**默认值**: `[]` |
| `reasoning_keep_policy` | 历史消息中 `reasoning_content` 字段的保留策略。<br><br>**类型**: `str`<br>**默认值**: `"never"` |
| `include_usage` | 是否在流式返回结果中包含 `usage` 信息。<br><br>**类型**: `bool`<br>**默认值**: `True` |

!!! info "补充"
    由于同一模型提供商的不同模型对 `tool_choice`、`response_format` 等参数的支持情况存在差异，这四个兼容性选项为类的**实例属性**。因此，创建对话模型类时可以传入值作为全局默认值（代表该提供商大部分模型支持的配置），后续如需针对特定模型进行微调，可在实例化时覆盖同名参数。

对于这些配置项的详细介绍如下：

??? note "1. supported_tool_choice"
    `tool_choice` 用于控制大模型在响应时是否以及调用哪个外部工具，以提升准确性、可靠性和可控性。常见的取值有：

    - `"auto"`：模型自主决定是否调用工具(默认行为)；
    - `"none"`：禁止调用工具；
    - `"required"`：强制调用至少一个工具；
    - 指定具体工具（在 OpenAI 兼容 API 中，具体为 `{"type": "function", "function": {"name": "xxx"}}`）。

    不同提供商支持范围不同。为避免错误，本库默认的`supported_tool_choice`为`["auto"]`，则在`bind_tools`时，`tool_choice`参数只能传递`auto`，如果传递其它取值均会被过滤。

    若需支持传递其它`tool_choice`取值，必须配置支持项。配置值为字符串列表，每个字符串的可选值：

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

    model = ChatVLLM(model="qwen3-4b")
    ```

    !!! info "提示"
        如无特殊需求，可保持默认（即`["auto"]`）。若业务场景要求模型**必须调用特定工具**或从**给定列表中任选其一**，且模型提供商支持对应策略，再按需开启：
        
        1. 如果要求**至少调用一个**工具，且模型提供商支持`required`，则可以设为 `["required"]`  （同时在调用`bind_tools`时，需要显示传递`tool_choice="required"`）

        2. 如果要求**调用指定**工具，且模型提供商支持指定某个具体的工具调用，则可以设为 `["specific"]`（在 `function_calling` 结构化输出中，此配置非常有用，可以确保模型调用指定的结构化输出工具，以保证结构化输出的稳定性。因为在 `with_structured_output` 方法中，其内部实现会在调用`bind_tools` 时传入**能够强制调用指定工具的 `tool_choice` 取值**，但如果 `supported_tool_choice` 中没有 `"specific"`，该参数将会被过滤。故如果想要保证能够正常传入 `tool_choice`，必须在 `supported_tool_choice` 中添加 `"specific"`。）

        该参数既可在创建时统一设置，也可在实例化时针对单模型动态覆盖；推荐在创建时一次性声明该提供商的大多数模型的`tool_choice`支持情况，而对于部分支持情况不同的模型，则在实例化时单独指定。

??? note "2. supported_response_format"
    目前常见的结构化输出方法有三种。

    - `function_calling`：通过调用一个符合指定 schema 的工具来生成结构化输出。
    - `json_schema`：由模型提供商提供的专门用于生成结构化输出的功能，在OpenAI兼容API中，具体为`response_format={"type": "json_schema", "json_schema": {...}}`。
    - `json_mode`：是某些提供商在推出`json_schema`之前提供的一种功能，它能生成有效的 JSON，但 schema 必须在提示（prompt）中进行描述。在 OpenAI 兼容 API 中，具体为 `response_format={"type": "json_object"}`）。

    其中，`json_schema` 仅少数 OpenAI 兼容 API 提供商支持（如 `OpenRouter`、`TogetherAI`）；`json_mode` 支持度更高，多数提供商已兼容；而 `function_calling` 最为通用，只要模型支持工具调用即可使用。

    本参数用于声明模型提供商对于`response_format`的支持情况。默认情况下为`[]`，代表模型提供商既不支持`json_mode`也不支持`json_schema`。此时`with_structured_output`方法中的`method`参数只能传递`function_calling`，如果传递了`json_mode`或`json_schema`，则会自动被转化为`function_calling`。如果想要启用`json_mode`或者`json_schema`的结构化输出实现方式，则需要显式设置该参数。

    例如vLLM部署的模型支持`json_schema`的结构化输出方法，则可以注册的时候进行声明：

    ```python
    from langchain_dev_utils.chat_models.adapters import create_openai_compatible_model

    ChatVLLM = create_openai_compatible_model(
        model_provider="vllm",
        chat_model_cls_name="ChatVLLM",
        compatibility_options={"supported_response_format": ["json_schema"]},
    )

    model = ChatVLLM(model="qwen3-4b")
    ``` 

    !!! info "提示"
        通常一般情况下也无需配置。仅在需要使用`with_structured_output`方法时需要考虑进行配置，此时，如果模型提供商支持`json_schema`，则可以考虑配置本参数（因为`json_schema`的结构化输出稳定性要优于`function_calling`）。以保证结构化输出的稳定性。对于`json_mode`，因为其只能保证输出JSON，因此一般没有必要设置。仅当模型不支持工具调用且仅支持设置`response_format={"type":"json_object"}`时，才需要配置本参数包含`json_mode`。
        
        同样，该参数既可在创建时统一设置，也可在实例化时针对单模型动态覆盖；推荐在创建时一次性声明该提供商的大多数模型的`response_format`支持情况，而对于部分支持情况不同的模型，则在实例化时单独指定。

    !!! warning "注意"
        本参数目前仅影响`model.with_structured_output`方法。对于`create_agent`中的结构化输出，若需要使用`json_schema`的实现方式，你需要确保对应模型的`profile`中包含`structured_output`字段，且值为`True`。

??? note "3. reasoning_keep_policy"

    用于控制历史消息（messages）中`reasoning_content` 字段的保留策略，主要适配于不同模型提供商的模型的不同的思考模式。

    支持以下取值：

    - `never`：在历史消息中**不保留任何**推理内容（默认)；

    - `current`：仅保留**当前对话**中的 `reasoning_content` 字段；

    - `all`：保留**所有对话**中的 `reasoning_content` 字段。

    例如：
    例如，用户先提问“纽约天气如何？”，随后追问“伦敦天气如何？”，当前正要进行第二轮对话，且即将进行最后一次模型调用。

    - 取值为`never`时

    当取值为`never`时，则最终传递给模型的 messages 中**不会有任何**的 `reasoning_content` 字段。最终模型收到的 messages 为：

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

    当取值为`current`时，仅保留**当前对话**中的 `reasoning_content` 字段。最终模型收到的 messages 为：
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

    当取值为`all`时，保留**所有**对话中的 `reasoning_content` 字段。最终模型收到的 messages 为：
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

    **注意**：如果本轮对话不涉及工具调用，则`current`效果和`never`效果相同。

    !!! info "提示"
        根据模型提供商对 `reasoning_content` 的保留要求灵活配置：

        - 若提供商要求**全程保留**推理内容，设为 `all`；  
        - 若仅要求在**本轮工具调用**中保留，设为 `current`；  
        - 若无特殊要求，保持默认 `never` 即可。  

        该参数既可在创建时统一设置，也可在实例化时针对单模型动态覆盖；由于同一提供商的不同模型对 `reasoning_content` 保留策略的可能不同，甚至同一模型在不同场景下可能需要不同的策略，**建议在实例化时显式指定**，创建类时无需赋值。

??? note "4. include_usage"

    控制是否在流式响应末尾附加 token 使用情况（`prompt_tokens` 和 `completion_tokens`）。

    对于OpenAI兼容的API，一般是通过 `stream_options={"include_usage": true}` 设置，因此只有支持 `stream_options` 参数的提供商才能使用。默认值为 `True`，如遇到不支持的提供商，或不希望记录 token 使用情况，可显式设为 `False`。


#### model_profiles参数设置

如果要使用 `model.profile` 参数，则必须在创建时显式传入。

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

!!! warning "注意"
    尽管已提供上述兼容性配置，本库仍无法保证 100% 适配所有 OpenAI 兼容接口。若模型提供商已有官方或社区集成类，请优先采用该集成类。如遇到任何兼容性问题，欢迎在本库 GitHub 仓库提交 issue。


### 创建嵌入模型类

与对话模型类类似，可以使用`create_openai_compatible_embedding`来创建嵌入模型类。

#### 示例代码
同样，我们使用`create_openai_compatible_embedding`来集成vLLM的嵌入模型。

!!! note "补充"  
    vLLM 可部署嵌入模型并暴露 OpenAI 兼容接口，例如：

    ```bash
    vllm serve Qwen/Qwen3-Embedding-4B \
    --task embed \
    --served-model-name qwen3-embedding-4b \
    --host 0.0.0.0 --port 8000
    ```

    服务地址为 `http://localhost:8000/v1`。


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

同样，`base_url`可以省略，此时需要设置环境变量 `VLLM_API_BASE`。

```bash
export VLLM_API_BASE="http://localhost:8000/v1"
```

代码处可以省略`base_url`。

```python
from langchain_dev_utils.embeddings.adapters import create_openai_compatible_embedding

VLLMEmbedding = create_openai_compatible_embedding(
    embedding_provider="vllm",
    embedding_model_cls_name="VLLMEmbedding",
)

embedding = VLLMEmbedding(model="qwen3-embedding-8b")

print(embedding.embed_query("你好"))
```

!!! warning "注意"
    与模型管理的要求类似，由于上述的两个函数底层使用了`pydantic.create_model`来创建模型类，因此会带来一定的性能开销，且`create_openai_compatible_model`底层也依托了一个全局字典来记录各个模型提供商对应的`profiles`，因此在使用中也存在多线程并发的问题，因此也建议在项目的启动阶段就创建好对应的集成类，后续不应再动态创建。

## 集成类的使用

### 对话模型类的使用

首先，我们需要创建对话模型类。我们使用之前创建好的`ChatVLLM`类。

- 支持`invoke`、`ainvoke`、`stream`、`astream`等方法。

??? example "普通调用"

    支持`invoke`进行简单的调用：

    ```python
    from langchain_core.messages import HumanMessage

    model = ChatVLLM("qwen3-4b")
    response = model.invoke([HumanMessage("Hello")])
    print(response)
    ```

    同时也支持`ainvoke`进行异步调用：

    ```python
    from langchain_core.messages import HumanMessage

    model = ChatVLLM("qwen3-4b")
    response = await model.ainvoke([HumanMessage("Hello")])
    print(response)
    ```

??? example "流式输出"

    支持`stream`进行流式输出：

    ```python
    from langchain_core.messages import HumanMessage

    model = ChatVLLM("qwen3-4b")
    for chunk in model.stream([HumanMessage("Hello")]):
        print(chunk)
    ```

    以及`astream`进行异步流式调用：

    ```python
    from langchain_core.messages import HumanMessage

    model = ChatVLLM("qwen3-4b")
    async for chunk in model.astream([HumanMessage("Hello")]):
        print(chunk)
    ```

- 支持`bind_tools`方法，进行工具调用。

如果模型本身支持工具调用，那么可以直接使用`bind_tools`方法进行工具调用：

??? example "工具调用"

    ```python
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    import datetime

    @tool
    def get_current_time() -> str:
        """获取当前时间戳"""
        return str(datetime.datetime.now().timestamp())

    model = ChatVLLM("qwen3-4b").bind_tools([get_current_time])
    response = model.invoke([HumanMessage("获取当前时间戳")])
    print(response)
    ```

- 支持`with_structured_output`方法，进行结构化输出。

如果该模型类的`supported_response_format`参数中包含`json_schema`，则`with_structured_output` 优先使用 `json_schema`进行结构化输出，否则回退 `function_calling`；如需 `json_mode`，显式指定 `method="json_mode"` 并确保注册时包含 `json_mode`。

??? example "结构化输出"

    ```python
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int

    model = ChatVLLM("qwen3-4b").with_structured_output(User)
    response = model.invoke([HumanMessage("你好，我叫张三，今年25岁")])
    print(response)
    ```


- 支持传递`BaseChatOpenAI`的参数，例如`temperature`、`top_p`、`max_tokens`等。

除此之外，由于该类继承了`BaseChatOpenAI`,因此支持传递`BaseChatOpenAI`的模型参数，例如`temperature`, `extra_body`等：

??? example "传递模型参数"

    ```python
    from langchain_core.messages import HumanMessage

    model = ChatVLLM("qwen3-4b",extra_body={"chat_template_kwargs": {"enable_thinking": False}}) #利用extra_body传递额外参数，这里是关闭思考模式
    response = model.invoke([HumanMessage("Hello")])
    print(response)
    ```


- 支持传递多模态数据

支持传递多模态数据，你可以使用 OpenAI 兼容的多模态数据格式或者直接使用`langchain`中的`content_block`。

??? example "传递多模态数据"

    **传递图片类数据**：

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

    model = ChatVLLM("qwen3-vl-2b")
    response = model.invoke(messages)
    print(response)
    ```

    **传递视频类数据**：
    

    ```python
    from langchain_core.messages import HumanMessage

    messages = [
        HumanMessage(
            content_blocks=[
                {
                    "type": "video",
                    "url": "https://example.com/video.mp4",
                },
                {"type": "text", "text": "描述这视频"},
            ]
        )
    ]

    model = ChatVLLM("qwen3-vl-2b")
    response = model.invoke(messages)
    print(response)
    ```
    
!!! note "补充"
    vllm 也支持部署多模态模型，例如 `qwen3-vl-2b`：
    ```bash
    vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3-vl-2b
    ```


- 支持 OpenAI 最新的`responses api` (暂未保证完全支持，可以用于简单测试，但不要用于生产环境)

该模型类也支持 OpenAI 最新的`responses_api`。但是目前仅有少量的提供商支持该风格的 API。如果你的模型提供商支持该 API 风格，则可以在传入`use_responses_api`参数为`True`。
    例如 vllm 支持`responses_api`，则可以这样使用：

??? example "OpenAI 最新的`responses_api`"

    ```python
    from langchain_core.messages import HumanMessage

    model = ChatVLLM("qwen3-4b", use_responses_api=True)
    response = model.invoke([HumanMessage(content="你好")])
    print(response)
    ```


### 嵌入模型类的使用

我们使用前面创建的`VLLMEmbeddings`类来初始化嵌入模型实例。

- 向量化查询

??? example "向量化查询"

    ```python
    embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
    print(embedding.embed_query("你好"))
    ```

    **异步写法**

    ```python
    embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
    res = await embedding.aembed_query("你好")
    print(res)
    ```

- 向量化字符串列表

??? example "向量化字符串列表"

    ```python
    documents = ["你好", "你好，我是张三"]
    embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
    print(embedding.embed_documents(documents))
    ```

    **异步写法**

    ```python
    documents = ["你好", "你好，我是张三"]
    embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
    res = await embedding.aembed_documents(documents)
    print(res)
    ```

**注意**：使用本功能创建的对话模型类和嵌入模型类支持传入任何`BaseChatOpenAI`和`OpenAIEmbeddings`的参数，例如`temperature`, `extra_body`,`dimensions`等。



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

