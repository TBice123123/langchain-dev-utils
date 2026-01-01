# Embeddings 模块 API 参考文档

## register_embeddings_provider

注册嵌入模型的提供者。

### 函数签名

```python
def register_embeddings_provider(
    provider_name: str,
    embeddings_model: EmbeddingsType,
    base_url: Optional[str] = None,
) -> None:
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| provider_name | str | 是 | - | 自定义提供者名称 |
| embeddings_model | EmbeddingsType | 是 | - | 嵌入模型类或支持的提供者字符串类型 |
| base_url | Optional[str] | 否 | None | 提供者的 BaseURL |

### 示例

```python
register_embeddings_provider("fakeembeddings", FakeEmbeddings)
register_embeddings_provider("vllm", "openai-compatible", base_url="http://localhost:8000/v1")
```

---

## batch_register_embeddings_provider

批量注册嵌入模型提供者。

### 函数签名

```python
def batch_register_embeddings_provider(
    providers: list[EmbeddingProvider]
) -> None:
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| providers | list[EmbeddingProvider] | 是 | - | 提供者配置列表 |

### 示例

```python
batch_register_embeddings_provider([
    {"provider_name": "fakeembeddings", "embeddings_model": FakeEmbeddings},
    {"provider_name": "vllm", "embeddings_model": "openai-compatible", "base_url": "http://localhost:8000/v1"},
])
```

---

## load_embeddings

从已注册的提供者加载嵌入模型。

### 函数签名

```python
def load_embeddings(
    model: str,
    *,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> Embeddings:
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| model | str | 是 | - | 模型名称，格式为 `model_name` 或 `provider_name:model_name` |
| provider | Optional[str] | 否 | None | 模型提供者名称 |
| **kwargs | Any | 否 | - | 额外的模型参数 |


### 示例

```python
embeddings = load_embeddings("vllm:qwen3-embedding-4b")
```

---

## create_openai_compatible_embedding

创建一个 OpenAI 兼容的嵌入模型类。

### 函数签名

```python
def create_openai_compatible_embedding(
    embedding_provider: str,
    base_url: Optional[str] = None,
    embedding_model_cls_name: Optional[str] = None,
) -> type[Embeddings]:
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| embedding_provider | str | 是 | - | 嵌入模型提供者名称 |
| base_url | Optional[str] | 否 | None | 模型提供者的 BaseURL |
| embedding_model_cls_name | Optional[str] | 否 | None | 自定义的嵌入模型类名 |

### 返回值

| 类型 | 描述 |
|------|------|
| type[Embeddings] | 动态创建的 OpenAI 兼容嵌入模型类 |


### 示例

```python
VLLMEmbeddings = create_openai_compatible_embedding(
    embedding_provider="vllm",
    base_url="http://localhost:8000/v1",
    embedding_model_cls_name="VLLMEmbeddings",
)
```
---

## EmbeddingsType

注册嵌入提供商时`embeddings_model`参数支持的类型。

### 类型定义

```python
EmbeddingsType = Union[type[Embeddings], Literal["openai-compatible"]]
```

---

## EmbeddingProvider

嵌入模型提供者配置类型。

### 类定义

```python
class EmbeddingProvider(TypedDict):
    provider_name: str
    embeddings_model: EmbeddingsType
    base_url: NotRequired[str]
```

### 字段说明

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| provider_name | str | 是 | 提供者名称 |
| embeddings_model | EmbeddingsType | 是 | 嵌入模型类或字符串 |
| base_url | NotRequired[str] | 否 | 基础 URL |