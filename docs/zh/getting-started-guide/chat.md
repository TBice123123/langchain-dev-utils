# 对话模型管理

## 概述

LangChain 的 `init_chat_model` 函数仅支持有限的模型提供商。本库提供更灵活的对话模型管理方案，支持自定义模型提供商，特别适用于需要接入未内置支持的模型服务（如 vLLM等）的场景。

## 注册模型提供商

注册对话模型提供商需调用 `register_model_provider`。对于不同的情况，注册步骤略有不同。

### 已有 LangChain 对话模型类

若模型提供商已有现成且合适的 LangChain 集成（详见[对话模型类集成](https://docs.langchain.com/oss/python/integrations/chat)），请将相应的集成对话模型类作为 chat_model 参数传入。

#### 代码示例

```python
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="fake_provider",
    chat_model=FakeChatModel,
)

# FakeChatModel仅用于测试，实际使用中必须传入具备真实功能的 ChatModel 类。
```

!!! tip "参数设置说明"
    `provider_name`代表模型提供商的名称，用于后续在 `load_chat_model` 中引用。命名必须以字母或数字开头，只能包含字母、数字和下划线，长度不超过 20 个字符。

#### 可选参数说明

**base_url**

此参数通常无需设置（因为对话模型类内部一般已定义默认的 API 地址），仅当需要覆盖对话模型类默认地址时才传入 `base_url`，且仅对字段名为 `api_base` 或 `base_url`（含别名）的属性生效。

**model_profiles**

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

这种情况下，需要`chat_model`参数必须设为 `"openai-compatible"`。

#### 代码示例

```python
register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1"
)
```

**注意**：关于这部分更多的细节，请参考[OpenAI 兼容 API 集成](../adavance-guide/openai-compatible/register.md)。


## 批量注册

若需注册多个提供商，可使用 `batch_register_model_provider` 避免重复调用。

#### 代码示例

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

    此外，注册时若将 `chat_model` 设为 `openai-compatible`，内部会通过 `pydantic.create_model` 动态创建新的模型类（以 `BaseChatOpenAICompatible` 为基类，生成对应的对话模型集成类），此过程涉及 Python 元类操作和 pydantic 验证逻辑初始化，存在一定性能开销，因此请避免在运行期频繁注册。


## 加载对话模型

使用 `load_chat_model` 函数加载对话模型（初始化对话模型实例）。

该函数接收 `model` 参数用于指定模型名称，可选的 `model_provider` 参数用于指定模型提供商；还可传入任意数量的关键字参数，用于传递对话模型类的额外参数。

#### 参数规则

- 若未传 `model_provider`，则 `model` 必须为 `provider_name:model_name` 格式；
- 若传 `model_provider`，则 `model` 必须仅为 `model_name`。

#### 代码示例

```python
# 方式一：model 包含 provider 信息
model = load_chat_model("vllm:qwen3-4b")

# 方式二：单独指定 provider
model = load_chat_model("qwen3-4b", model_provider="vllm")
```

### 模型方法和参数

对于支持的模型方法和参数，需要参考对应的对话模型类的使用说明。如果采用的是第二种情况，则支持所有的`BaseChatOpenAI` 类的方法和参数。

### 兼容官方提供商

对于 LangChain 官方已支持的提供商（如 `openai`），可直接使用 `load_chat_model` 无需注册：

```python
model = load_chat_model("openai:gpt-4o-mini")
# 或
model = load_chat_model("gpt-4o-mini", model_provider="openai")
```

!!! success "最佳实践"
    对于本模块的使用，可以根据下面三种情况进行选择：

    1. 若接入的所有模型提供商均被官方 `init_chat_model` 支持，请直接使用官方函数，以获得最佳兼容性和稳定性。

    2. 若接入的部分模型提供商为非官方支持，可使用本模块的功能，先利用`register_model_provider`注册模型提供商，然后使用`load_chat_model`加载模型。

    3. 若接入的模型提供商暂无适合的集成，但提供商提供了 OpenAI 兼容的 API（如 vLLM），则推荐使用本模块的功能，先利用`register_model_provider`注册模型提供商（chat_model传入`openai-compatible`），然后使用`load_chat_model`加载模型。
