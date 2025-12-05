# Embeddings 模块的 API 参考

## register_embeddings_provider

注册嵌入模型的提供者。

```python
def register_embeddings_provider(
    provider_name: str,
    embeddings_model: EmbeddingsType,
    base_url: Optional[str] = None,
)-> None:
```

**参数说明：**

- `provider_name`：字符串类型，必填，自定义提供者名称
- `embeddings_model`：嵌入模型类或支持的提供者字符串类型，必填
- `base_url`：可选字符串类型，提供者的 BaseURL

**示例：**

```python
register_embeddings_provider("fakeembeddings", FakeEmbeddings)
register_embeddings_provider("vllm", "openai-compatible", base_url="http://localhost:8000/v1")
```

## batch_register_embeddings_provider

批量注册嵌入模型提供者。

```python
def batch_register_embeddings_provider(
    providers: list[EmbeddingProvider]
)-> None:
```

**参数说明：**

- `providers`：EmbeddingProvider 列表类型，必填，提供者配置列表

**示例：**

```python
batch_register_embeddings_provider([
    {"provider_name": "fakeembeddings", "embeddings_model": FakeEmbeddings},
    {"provider_name": "vllm", "embeddings_model": "openai-compatible", "base_url": "http://localhost:8000/v1"},
])
```

## load_embeddings

从已注册的提供者加载嵌入模型。

```python
def load_embeddings(
    model: str,
    *,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> Embeddings:
```

**参数说明：**

- `model`：字符串类型，必填，模型名称，格式为 `model_name` 或 `provider_name:model_name`
- `provider`：可选字符串类型，模型提供者名称
- `**kwargs`：任意类型，可选，额外的模型参数

**返回值：** Embeddings 类型，加载的嵌入模型实例

**示例：**

```python
embeddings = load_embeddings("vllm:qwen3-embedding-4b")
```

## EmbeddingsType

注册嵌入提供商时`embeddings_model`参数支持的类型。

```python
EmbeddingsType = Union[type[Embeddings], Literal["openai-compatible"]]
```

## EmbeddingProvider

嵌入模型提供者配置类型。

```python
class EmbeddingProvider(TypedDict):
    provider_name: str
    embeddings_model: EmbeddingsType
    base_url: NotRequired[str]
```

**字段说明：**

- `provider_name`：字符串类型，必填，提供者名称
- `embeddings_model`：Embeddings 类型或字符串类型，必填，嵌入模型类或字符串
- `base_url`：非必需字符串类型，基础 URL
