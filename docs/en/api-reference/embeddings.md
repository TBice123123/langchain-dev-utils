# Embeddings Module API Reference Documentation

## register_embeddings_provider

Register a provider for embedding models.

### Function Signature

```python
def register_embeddings_provider(
    provider_name: str,
    embeddings_model: EmbeddingsType,
    base_url: Optional[str] = None,
) -> None:
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| provider_name | str | Yes | - | Custom provider name |
| embeddings_model | EmbeddingsType | Yes | - | Embedding model class or supported provider string type |
| base_url | Optional[str] | No | None | BaseURL of the provider |

### Example

```python
register_embeddings_provider("fakeembeddings", FakeEmbeddings)
register_embeddings_provider("vllm", "openai-compatible", base_url="http://localhost:8000/v1")
```

---

## batch_register_embeddings_provider

Batch register embedding model providers.

### Function Signature

```python
def batch_register_embeddings_provider(
    providers: list[EmbeddingProvider]
) -> None:
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| providers | list[EmbeddingProvider] | Yes | - | List of provider configurations |

### Example

```python
batch_register_embeddings_provider([
    {"provider_name": "fakeembeddings", "embeddings_model": FakeEmbeddings},
    {"provider_name": "vllm", "embeddings_model": "openai-compatible", "base_url": "http://localhost:8000/v1"},
])
```

---

## load_embeddings

Load an embedding model from registered providers.

### Function Signature

```python
def load_embeddings(
    model: str,
    *,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> Embeddings:
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| model | str | Yes | - | Model name, format: `model_name` or `provider_name:model_name` |
| provider | Optional[str] | No | None | Model provider name |
| **kwargs | Any | No | - | Additional model parameters |

### Example

```python
embeddings = load_embeddings("vllm:qwen3-embedding-4b")
```

---

## create_openai_compatible_embedding

Create an OpenAI compatible embedding model class.

### Function Signature

```python
def create_openai_compatible_embedding(
    embedding_provider: str,
    base_url: Optional[str] = None,
    embedding_model_cls_name: Optional[str] = None,
) -> type[Embeddings]:
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| embedding_provider | str | Yes | - | Embedding model provider name |
| base_url | Optional[str] | No | None | BaseURL of the model provider |
| embedding_model_cls_name | Optional[str] | No | None | Custom embedding model class name |

### Return Value

| Type | Description |
|------|-------------|
| type[Embeddings] | Dynamically created OpenAI compatible embedding model class |

### Example

```python
VLLMEmbeddings = create_openai_compatible_embedding(
    embedding_provider="vllm",
    base_url="http://localhost:8000/v1",
    embedding_model_cls_name="VLLMEmbeddings",
)
```
---

## EmbeddingsType

Types supported by the `embeddings_model` parameter when registering embedding providers.

### Type Definition

```python
EmbeddingsType = Union[type[Embeddings], Literal["openai-compatible"]]
```

---

## EmbeddingProvider

Embedding model provider configuration type.

### Class Definition

```python
class EmbeddingProvider(TypedDict):
    provider_name: str
    embeddings_model: EmbeddingsType
    base_url: NotRequired[str]
```

### Field Description

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| provider_name | str | Yes | Provider name |
| embeddings_model | EmbeddingsType | Yes | Embedding model class or string |
| base_url | NotRequired[str] | No | Base URL |