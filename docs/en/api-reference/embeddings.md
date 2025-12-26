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
|------|------|------|--------|------|
| provider_name | str | Yes | - | Custom provider name |
| embeddings_model | EmbeddingsType | Yes | - | Embedding model class or supported provider string type |
| base_url | Optional[str] | No | None | Base URL of the provider |

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
|------|------|------|--------|------|
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
|------|------|------|--------|------|
| model | str | Yes | - | Model name, format as `model_name` or `provider_name:model_name` |
| provider | Optional[str] | No | None | Model provider name |
| **kwargs | Any | No | - | Additional model parameters |

### Example

```python
embeddings = load_embeddings("vllm:qwen3-embedding-4b")
```

---

## EmbeddingsType

Types supported for the `embeddings_model` parameter when registering an embedding provider.

### Type Definition

```python
EmbeddingsType = Union[type[Embeddings], Literal["openai-compatible"]]
```

---

## EmbeddingProvider

Configuration type for embedding model providers.

### Class Definition

```python
class EmbeddingProvider(TypedDict):
    provider_name: str
    embeddings_model: EmbeddingsType
    base_url: NotRequired[str]
```

### Field Description

| Field | Type | Required | Description |
|------|------|------|------|
| provider_name | str | Yes | Provider name |
| embeddings_model | EmbeddingsType | Yes | Embedding model class or string |
| base_url | NotRequired[str] | No | Base URL |