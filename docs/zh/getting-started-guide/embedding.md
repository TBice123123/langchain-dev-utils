# 嵌入模型管理

## 概述

LangChain 的 `init_embeddings` 函数仅支持有限的嵌入模型提供商。本库提供更灵活的嵌入模型管理方案，特别适用于需要接入未内置支持的嵌入服务（如 vLLM 等）的场景。


## 注册嵌入模型提供商

注册嵌入模型提供商需调用 `register_embeddings_provider`。根据 `embeddings_model` 类型不同，注册方式略有差异。

### 已有 LangChain 嵌入模型类

若嵌入模型提供商已有现成且合适的 LangChain 集成（详见 [嵌入模型集成列表](https://docs.langchain.com/oss/python/integrations/text_embedding)），请将相应的嵌入模型类直接传入 `embeddings_model` 参数。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `provider_name` | 模型提供商名称，用于后续在 `load_embeddings` 中引用。<br><br>**类型**: `str`<br>**必填**: 是 |
| `embeddings_model` | LangChain 嵌入模型类。<br><br>**类型**: `type[Embeddings]`<br>**必填**: 是 |
| `base_url` | API 基础地址，通常无需手动设置。<br><br>**类型**: `str`<br>**必填**: 否 |

#### 代码示例

```python
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_dev_utils.embeddings import register_embeddings_provider

register_embeddings_provider(
    provider_name="fake_provider",
    embeddings_model=FakeEmbeddings,
)
```

#### 使用说明

- `FakeEmbeddings` 仅用于测试。实际使用中必须传入具备真实功能的 `Embeddings` 类。
- `provider_name` 代表模型提供商的名称，用于后续在 `load_embeddings` 中引用。名称可自定义，但不要包含 `:`、`-` 等特殊字符。

!!! warning "注意"
    `provider_name` 必须以字母或数字开头，只能包含字母、数字和下划线，长度不超过 20 个字符。


#### 可选参数说明

**base_url**

此参数通常无需设置（因为嵌入模型类内部一般已定义默认的 API 地址），仅当需要覆盖嵌入模型类默认地址时才传入 `base_url`，且仅对字段名为 `api_base` 或 `base_url`（含别名）的属性生效。

### 未有 LangChain 嵌入模型类，但提供商支持 OpenAI 兼容 API

这种情况下的参数说明如下：

#### 参数说明

| 参数 | 说明 |
|------|------|
| `provider_name` | 模型提供商名称，用于后续在 `load_embeddings` 中引用。<br><br>**类型**: `str`<br>**必填**: 是 |
| `embeddings_model` | 固定取值 `"openai-compatible"`。<br><br>**类型**: `str`<br>**必填**: 是 |
| `base_url` | API 基础地址。<br><br>**类型**: `str`<br>**必填**: 否 |



#### 代码示例

**方式一：显式传参**

```python
register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible",
    base_url="http://localhost:8000/v1"
)
```

**方式二：环境变量（推荐）**

```bash
export VLLM_API_BASE=http://localhost:8000/v1
```

```python
register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible"
    # 自动读取 VLLM_API_BASE
)
```

**注意**：关于这部分更多的细节，请参考[OpenAI 兼容 API 集成](../adavance-guide/openai-compatible.md)。


## 批量注册

若需注册多个提供商，可使用 `batch_register_embeddings_provider`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `providers` | 提供商配置列表，每个字典包含注册参数。<br><br>**类型**: `list[dict]`<br>**必填**: 是 |

#### 代码示例

```python
from langchain_dev_utils.embeddings import batch_register_embeddings_provider
from langchain_core.embeddings.fake import FakeEmbeddings

batch_register_embeddings_provider(
    providers=[
        {
            "provider_name": "fake_provider",
            "embeddings_model": FakeEmbeddings,
        },
        {
            "provider_name": "vllm",
            "embeddings_model": "openai-compatible",
            "base_url": "http://localhost:8000/v1",
        },
    ]
)
```

!!! warning "注意"
    两个注册函数均基于全局字典实现。**必须在应用启动阶段完成所有注册**，禁止运行时动态注册，以避免多线程问题。 
     
    此外，注册时若将 `embeddings_model` 设为 `openai-compatible`，内部会通过 `pydantic.create_model` 动态创建新的模型类（以 `BaseEmbeddingOpenAICompatible` 为基类，生成对应的嵌入模型集成类），此过程涉及 Python 元类操作和 pydantic 验证逻辑初始化，存在一定性能开销，因此请避免在运行期频繁注册。
    

## 加载嵌入模型

使用 `load_embeddings` 初始化嵌入模型实例。

#### 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | `str` | 是 | - | 模型名称 |
| `provider` | `str` | 否 | `None` | 模型提供商名称 |

**除此之外，还可以传入任意数量的关键字参数，用于传递嵌入模型类的额外参数。**

#### 参数规则

- 若未传 `provider`，则 `model` 必须为 `provider_name:embeddings_name` 格式；
- 若传 `provider`，则 `model` 仅为 `embeddings_name`。

#### 代码示例

```python
# 方式一：model 包含 provider 信息
embedding = load_embeddings("vllm:qwen3-embedding-4b")

# 方式二：单独指定 provider
embedding = load_embeddings("qwen3-embedding-4b", provider="vllm")
```

## 模型方法和参数

对于支持的模型方法和参数，需要参考对应的嵌入模型类的使用说明。如果采用的是第二种情况，则支持所有的`OpenAIEmbeddings` 类的方法和参数。


### 兼容官方提供商

对于 LangChain 官方已支持的提供商（如 `openai`），可直接使用 `load_embeddings` 无需注册：

```python
model = load_embeddings("openai:text-embedding-3-large")
# 或
model = load_embeddings("text-embedding-3-large", provider="openai")
```

!!! success "最佳实践"
    对于本模块的使用，可以根据下面三种情况进行选择：

    1. 若接入的所有嵌入模型提供商均被官方 `init_embeddings` 支持，请直接使用官方函数，以获得最佳兼容性。

    2. 若接入的部分嵌入模型提供商为非官方支持，可利用本模块的注册与加载机制，先利用`register_embeddings_provider`注册模型提供商，然后使用`load_embeddings`加载模型。

    3. 若接入的嵌入模型提供商暂无适合的集成，但提供商提供了 OpenAI 兼容的 API（如 vLLM），则推荐利用本模块的功能，先利用`register_embeddings_provider`注册模型提供商（embeddings_model传入`openai-compatible`），然后使用`load_embeddings`加载模型。
