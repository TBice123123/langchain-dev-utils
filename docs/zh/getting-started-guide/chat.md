# 对话模型管理

## 概述

LangChain 的 `init_chat_model` 函数仅支持有限的模型提供商。本库提供更灵活的对话模型管理方案，支持自定义模型提供商，特别适用于需要接入未内置支持的模型服务（如 vLLM、OpenRouter 等）的场景。

## 注册模型提供商

注册对话模型提供商需调用 `register_model_provider`。对于不同的情况，注册步骤略有不同。

### 已有 LangChain 对话模型类

若模型提供商已有现成且合适的 LangChain 集成（详见[对话模型类集成](https://docs.langchain.com/oss/python/integrations/chat)），请将相应的集成对话模型类作为 chat_model 参数传入。

具体代码可参考如下：
```python
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="fake_provider",
    chat_model=FakeChatModel,
)
```
对于上述代码有以下几点补充：

- `FakeChatModel` 仅用于测试。实际使用中必须传入具备真实功能的 `ChatModel` 类。
- `provider_name` 代表模型提供商的名称，用于后续在 `load_chat_model` 中引用。名称可自定义，但不要包含":"、"-"等特殊字符。

除此之外，本情况下，该函数还接收以下参数：

- **base_url**

**此参数通常无需设置（因为模型类内部一般已定义默认的API 地址）**，仅当需要覆盖模型类默认地址时才传入 `base_url`，且仅对字段名为 `api_base` 或 `base_url`（含别名）的属性生效。

- **model_profiles**

如果你的 LangChain 集成对话模型类已全面支持 `profile` 参数（即可以通过 `model.profile` 直接访问模型的相关属性，例如 `max_input_tokens`、`tool_calling` 等），则无需额外设置 `model_profiles`。

如果通过 `model.profile` 访问时返回的是一个空字典 `{}`，说明该 LangChain 对话模型类可能暂时未支持 `profile` 参数，此时可以手动提供 `model_profiles`。

`model_profiles` 是一个字典，其每一个键为模型名称，值为对应模型的 profile 配置:

```python
{
    "model_name_1": {
        "max_input_tokens": 100_000,
        "tool_calling": True,
        "structured_output": True,
        # ... 其他可选字段
    },
    "model_name_2": {
        "max_input_tokens": 32768,
        "image_inputs": True,
        "tool_calling": False,
        # ... 其他可选字段
    },
    # 可以有任意多个模型配置
}
```
!!! info "提示"
    推荐使用 `langchain-model-profiles` 库来获取你所用模型提供商的 profiles。

### 未有 LangChain 对话模型类，但模型提供商支持 OpenAI 兼容 API


很多模型提供商都支持 **OpenAI 兼容 API** 的服务，例如：[vLLM](https://github.com/vllm-project/vllm)、[OpenRouter](https://openrouter.ai/)、[Together AI](https://www.together.ai/)等。当你接入的模型提供商未有合适的 LangChain 对话模型类时，但提供商支持 OpenAI 兼容 API 时，可以考虑使用此情况。

本库会根据用户的相关输入，使用内置 `BaseChatOpenAICompatible` 类构建对应于特定提供商的对话模型类。该类继承自 `langchain-openai` 的 `BaseChatOpenAI`，并增强以下能力：

- **支持更多格式的推理内容**： 相较于`ChatOpenAI` 只能输出官方的推理内容，本类还支持输出更多格式的推理内容（例如`vLLM`）。
- **动态适配并选择最合适的结构化输出方法**：默认情况下，能够根据模型提供商的实际支持情况，自动选择最优的结构化输出方法（`function_calling` 或 `json_schema`）。  
- **通过 compatibility_options 精细适配差异**： 通过配置提供商兼容性选项，解决`tool_choice`、`response_format` 等参数的支持差异。

**注意**：使用此情况时，必须安装 standard 版本的 `langchain-dev-utils` 库。具体可以参考安装部分的介绍。

此情况下，除了要传入 `provider_name`以及`chat_model`（取值必须为`"openai-compatible"`）参数外，还需传入 `base_url` 参数。

对于`base_url`参数，可通过以下任一方式提供：

  - **显式传参**：
    ```python
    register_model_provider(
        provider_name="vllm",
        chat_model="openai-compatible",
        base_url="http://localhost:8000/v1"
    )
    ```
  - **通过环境变量**（推荐用于配置管理）：
    ```bash
    export VLLM_API_BASE=http://localhost:8000/v1
    ```
    代码中可省略 `base_url`：
    ```python
    register_model_provider(
        provider_name="vllm",
        chat_model="openai-compatible"
        # 自动读取 VLLM_API_BASE
    )
    ``` 
!!! info "提示"
    此情况下，模型提供商的 API 端点的环境变量的命名规则是`${PROVIDER_NAME}_API_BASE`（全大写，下划线分隔）。对应的 API_KEY 环境变量的命名规则是`${PROVIDER_NAME}_API_KEY`（全大写，下划线分隔）。



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


同时，本情况下，还可以设置以下两个可选参数：

- **model_profiles**

这种情况下，若未手动设置 `model_profiles`，则 `model.profile` 将返回一个空字典 `{}`。因此，若需通过 `model.profile` 获取指定模型的配置信息，必须先显式设置 `model_profiles`。


- **compatibility_options**

仅在此情况下有效。用于**声明**该提供商对**OpenAI API**的部分特性的支持情况，以提高兼容性和稳定性。
目前支持以下配置项：

- `supported_tool_choice`：支持的 `tool_choice` 策略列表，默认为`["auto"]`；
- `supported_response_format`：支持的 `response_format` 格式列表(`json_schema`、`json_object`)，默认为 `[]`；
- `reasoning_keep_policy`：传给模型的历史消息（messages）中 `reasoning_content` 字段的保留策略。可选值有`never`、`current`、`all`。默认为`never`。
- `include_usage`：是否在最后一条流式返回结果中包含 `usage` 信息，默认为 `True`。

!!! info "补充"
    因为同一模型提供商的不同模型对于`tool_choice`、`response_format`等参数的支持情况有所差异。故该四个兼容性选项最终会作为该类的**实例属性**。注册时可传值作为全局默认值（代表该提供商的大部分模型支持的配置），加载时如需微调，在 `load_chat_model` 中覆盖同名参数即可。


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
    register_model_provider(
        provider_name="vllm",
        chat_model="openai-compatible",
        compatibility_options={"supported_tool_choice": ["auto", "none", "required", "specific"]},
    )
    ```

    !!! info "提示"
        如无特殊需求，可保持默认（即`["auto"]`）。若业务场景要求模型**必须调用特定工具**或从**给定列表中任选其一**，且模型提供商支持对应策略，再按需开启：
        
        1. 如果要求**至少调用一个**工具，且模型提供商支持`required`，则可以设为 `["required"]`  （同时在调用`bind_tools`时，需要显示传递`tool_choice="required"`）

        2. 如果要求**调用指定**工具，且模型提供商支持指定某个具体的工具调用，则可以设为 `["specific"]`（在 `function_calling` 结构化输出中，此配置非常有用，可以确保模型调用指定的结构化输出工具，以保证结构化输出的稳定性。因为在 `with_structured_output` 方法中，其内部实现会在调用`bind_tools` 时传入**能够强制调用指定工具的 `tool_choice` 取值**，但如果 `supported_tool_choice` 中没有 `"specific"`，该参数将会被过滤。故如果想要保证能够正常传入 `tool_choice`，必须在 `supported_tool_choice` 中添加 `"specific"`。）

        该参数既可在 `register_model_provider` 中统一设置，也可在 `load_chat_model` 时针对单模型动态覆盖；推荐在 `register_model_provider` 中一次性声明该提供商的大多数模型的`tool_choice`支持情况，而对于部分支持情况不同的模型，则在 `load_chat_model` 中单独指定。

??? note "2. supported_response_format"
    目前常见的结构化输出方法有三种。

    - `function_calling`：通过调用一个符合指定 schema 的工具来生成结构化输出。
    - `json_schema`：由模型提供商提供的专门用于生成结构化输出的功能，在OpenAI兼容API中，具体为`response_format={"type": "json_schema", "json_schema": {...}}`。
    - `json_mode`：是某些提供商在推出`json_schema`之前提供的一种功能，它能生成有效的 JSON，但 schema 必须在提示（prompt）中进行描述。在 OpenAI 兼容 API 中，具体为 `response_format={"type": "json_object"}`）。

    其中，`json_schema` 仅少数 OpenAI 兼容 API 提供商支持（如 `OpenRouter`、`TogetherAI`）；`json_mode` 支持度更高，多数提供商已兼容；而 `function_calling` 最为通用，只要模型支持工具调用即可使用。

    本参数用于声明模型提供商对于`response_format`的支持情况。默认情况下为`[]`，代表模型提供商既不支持`json_mode`也不支持`json_schema`。此时`with_structured_output`方法中的`method`参数只能传递`function_calling`(或者是`auto`,此时`auto`将被推断为`function_calling`），如果传递了`json_mode`或`json_schema`，则会自动被转化为`function_calling`。如果想要启用`json_mode`或者`json_schema`的结构化输出实现方式，则需要显示设置该参数。

    例如OpenRouter大多数模型同时支持`json_mode`和`json_schema`的`response_format`，则可以注册的时候进行声明：

    ```python
    register_model_provider(
        provider_name="openrouter",
        chat_model="openai-compatible",
        compatibility_options={"supported_response_format": ["json_mode", "json_schema"]},
    )
    ``` 

    !!! info "提示"
        通常一般情况下也无需配置。仅在需要使用`with_structured_output`方法时需要考虑进行配置，此时，如果模型提供商支持`json_schema`，则可以考虑配置本参数。以保证结构化输出的稳定性。对于`json_mode`，因为其只能保证输出JSON，因此一般没有必要设置。仅当模型不支持工具调用且仅支持设置`response_format={"type":"json_object"}`时，才需要配置本参数包含`json_mode`。
        
        同样，该参数既可在 `register_model_provider` 中统一设置，也可在 `load_chat_model` 时针对单模型动态覆盖；推荐在 `register_model_provider` 中一次性声明该提供商的大多数模型的`response_format`支持情况，而对于部分支持情况不同的模型，则在 `load_chat_model` 中单独指定。

    !!! warning "注意"
        本参数目前仅影响`model.with_structured_output`方法。对于`create_agent`中的结构化输出，若需要使用`json_schema`的实现方式，你需要确保对应模型的`profile`中包含`structured_output`字段，且值为`True`。

??? note "3. reasoning_keep_policy"

    用于控制历史消息（messages）中`reasoning_content` 字段的保留策略。

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

        同样，该参数既可在 `register_model_provider` 中统一设置，也可在 `load_chat_model` 时针对单模型动态覆盖；若需要保留`reasoning_content`的情况的模型较少，推荐在 `load_chat_model` 中单独指定，此时`register_model_provider`无需设置。



??? note "4. include_usage"

    `include_usage` 是 OpenAI 兼容 API 中的一个参数，用于控制是否在流式响应的末尾附加一条包含 token 使用情况（如 `prompt_tokens` 和 `completion_tokens`）的消息。由于标准流式响应默认不返回用量信息，启用该选项后，客户端可直接获取完整的 token 消耗数据，便于计费、监控或日志记录。

    通常通过 `stream_options={"include_usage": true}` 启用。考虑到有些模型提供商不支持该参数，因此本库将其设为兼容性选项，默认值为 `True`，因为绝大多数模型提供商均支持该参数，如果不支持，则可以显式设为 `False`。

    !!! info "提示"
        此参数一般无需设置，保持默认值即可。只有在模型提供商不支持时，才需要设置为 `False`。



!!! warning "注意"
    尽管已提供上述兼容性配置，本库仍无法保证 100% 适配所有 OpenAI 兼容接口。若模型提供商已有官方或社区集成类，请优先采用该集成类。如遇到任何兼容性问题，欢迎在本库 GitHub 仓库提交 issue。


## 批量注册

若需注册多个提供商，可使用 `batch_register_model_provider` 避免重复调用。

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

!!! warning "注意"
    两个注册函数均基于全局字典实现。为避免多线程问题，**必须在应用启动阶段完成所有注册**，禁止运行时动态注册。  


## 加载对话模型

使用 `load_chat_model` 函数加载对话模型（初始化对话模型实例）。其中，参数的规则如下：

- 若未传 `model_provider`，则 `model` 必须为 `provider_name:model_name` 格式；
- 若传 `model_provider`，则 `model` 必须仅为 `model_name`。

**示例**：

```python
# 方式一
model = load_chat_model("vllm:qwen3-4b")

# 方式二
model = load_chat_model("qwen3-4b", model_provider="vllm")
```

虽然`vLLM` 不强制需要 API Key，但 LangChain 仍要求设置。可以在环境变量中设置：

```bash
export VLLM_API_KEY=vllm
```

### 模型方法和参数

对于**情况一**，其所有方法与参数均与该对应的对话模型类保持一致。  
而对于**情况二**，则模型的方法和参数如下：

- 支持`invoke`、`ainvoke`、`stream`、`astream`等方法。
- 支持`bind_tools`方法，进行工具调用。
- 支持`with_structured_output`方法，进行结构化输出。
- 支持传递`BaseChatOpenAI`的参数，例如`temperature`、`top_p`、`max_tokens`等。
- 支持传递多模态数据
- 支持 OpenAI 最新的`responses api` (暂未保证完全支持，可以用于简单测试，但不要用于生产环境)

??? note "普通调用"

    支持`invoke`进行简单的调用：

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage

    model = load_chat_model("vllm:qwen3-4b")
    response = model.invoke([HumanMessage("Hello")])
    print(response)
    ```

    同时也支持`ainvoke`进行异步调用：

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage

    model = load_chat_model("vllm:qwen3-4b")
    response = await model.ainvoke([HumanMessage("Hello")])
    print(response)
    ```

??? note "流式输出"

    支持`stream`进行流式输出：

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage

    model = load_chat_model("vllm:qwen3-4b")
    for chunk in model.stream([HumanMessage("Hello")]):
        print(chunk)
    ```

    以及`astream`进行异步流式调用：

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage

    model = load_chat_model("vllm:qwen3-4b")
    async for chunk in model.astream([HumanMessage("Hello")]):
        print(chunk)
    ```

??? note "工具调用"

    如果模型本身支持工具调用，那么可以直接使用`bind_tools`方法进行工具调用：

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    import datetime

    @tool
    def get_current_time() -> str:
        """获取当前时间戳"""
        return str(datetime.datetime.now().timestamp())

    model = load_chat_model("vllm:qwen3-4b").bind_tools([get_current_time])
    response = model.invoke([HumanMessage("获取当前时间戳")])
    print(response)
    ```


??? note "结构化输出"

    支持结构化输出，默认的`method`取值为`auto`，此时将会根据模型提供商的`supported_response_format`参数自动选择合适的结构化输出方法。具体为如果其中取值包含`json_schema`，则会选择`json_schema`方法；否则，则会选择`function_calling`方法。

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int

    model = load_chat_model("vllm:qwen3-4b").with_structured_output(User)
    response = model.invoke([HumanMessage("你好，我叫张三，今年25岁")])
    print(response)
    ```
    相较于工具调用，`json_schema`能100%保证输出符合JSON Schema规范，避免工具调用可能产生的参数误差。故如果模型提供商支持`json_schema`，则默认会采用此方法。当模型提供商不支持时，才会回退到`function_calling`方法。
    对于`json_mode`，虽然支持度较高，但是由于其必须在提示词中引导模型输出指定Schema的JSON字符串，因此使用起来比较麻烦，故默认不采用此方法。如想要采用，则可以显示提供`method="json_mode"`（前提是注册或者实例化的时候保证`supported_response_format`取值中包含`json_mode`)。

??? note "传递模型参数"

    除此之外，由于该类继承了`BaseChatOpenAI`,因此支持传递`BaseChatOpenAI`的模型参数，例如`temperature`, `extra_body`等：

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage

    model = load_chat_model("vllm:qwen3-4b",extra_body={"chat_template_kwargs": {"enable_thinking": False}}) #利用extra_body传递额外参数，这里是关闭思考模式
    response = model.invoke([HumanMessage("Hello")])
    print(response)
    ```


??? note "传递多模态数据"

    支持传递多模态数据，你可以使用 OpenAI 兼容的多模态数据格式或者直接使用`langchain`中的`content_block`。例如：

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
                {"type": "text", "text": "描述这张图片"},
            ]
        )
    ]

    model = load_chat_model("vllm:qwen3-vl-2b")
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


??? note "OpenAI 最新的`responses_api`"

    该模型类也支持 OpenAI 最新的`responses_api`。但是目前仅有少量的提供商支持该风格的 API。如果你的模型提供商支持该 API 风格，则可以在传入`use_responses_api`参数为`True`。
    例如 vllm 支持`responses_api`，则可以这样使用：

    ```python
    from langchain_dev_utils.chat_models import load_chat_model
    from langchain_core.messages import HumanMessage

    model = load_chat_model("vllm:qwen3-4b", use_responses_api=True)
    response = model.invoke([HumanMessage(content="你好")])
    print(response)
    ```

!!! info "提示"
    无论是哪种情况，你都可以传入任意数量的关键字参数作为模型的额外参数，例如`temperature`, `extra_body`等。

### 兼容官方提供商

对于 LangChain 官方已支持的提供商（如 `openai`），可直接使用 `load_chat_model` 无需注册：

```python
model = load_chat_model("openai:gpt-4o-mini")
# 或
model = load_chat_model("gpt-4o-mini", model_provider="openai")
```

!!! note "最佳实践"
    对于本模块的使用，可以根据下面三种情况进行选择：

    1. 若接入的所有模型提供商均被官方 `init_chat_model` 支持，请直接使用官方函数，以获得最佳兼容性和稳定性。

    2. 若接入的部分模型提供商为非官方支持，可使用本模块的功能，先利用`register_model_provider`注册模型提供商，然后使用`load_chat_model`加载模型。

    3. 若接入的模型提供商暂无适合的集成，但提供商提供了 OpenAI 兼容的 API（如 vLLM、OpenRouter），则推荐使用本模块的功能，先利用`register_model_provider`注册模型提供商（chat_model传入`openai-compatible`），然后使用`load_chat_model`加载模型。
