# Embeddings Module API Reference

## register_embeddings_provider

Registers a provider for embeddings models.

```python
def register_embeddings_provider(
    provider_name: str,
    embeddings_model: EmbeddingsType,
    base_url: Optional[str] = None,
) -> None:
```

**Parameters:**

- `provider_name`: string, required, custom provider name
- `embeddings_model`: embeddings model class or supported provider string type, required
- `base_url`: optional string, provider's BaseURL

**Examples:**

```python
register_embeddings_provider("fakeembeddings", FakeEmbeddings)
register_embeddings_provider("vllm", "openai-compatible", base_url="http://localhost:8000/v1")
```

## batch_register_embeddings_provider

Batch registers embeddings model providers.

```python
def batch_register_embeddings_provider(
    providers: list[EmbeddingProvider]
) -> None:
```

**Parameters:**

- `providers`: EmbeddingProvider list type, required, list of provider configurations

**Examples:**

```python
batch_register_embeddings_provider([
    {"provider_name": "fakeembeddings", "embeddings_model": FakeEmbeddings},
    {"provider_name": "vllm", "embeddings_model": "openai-compatible", "base_url": "http://localhost:8000/v1"},
])
```

## load_embeddings

Loads an embeddings model from a registered provider.

```python
def load_embeddings(
    model: str,
    *,
    model_provider: Optional[str] = None,
    **kwargs: Any,
) -> Embeddings:
```

**Parameters:**

- `model`: string, required, model name in format `model_name` or `provider_name:model_name`
- `model_provider`: optional string, model provider name
- `**kwargs`: any type, optional, additional model parameters

**Returns:** Embeddings type, loaded embeddings model instance

**Examples:**

```python
embeddings = load_embeddings("vllm:qwen3-embedding-4b")
```

## EmbeddingsType

Supported types for the `embeddings_model` parameter when registering an embeddings provider.

```python
EmbeddingsType = Union[type[Embeddings], Literal["openai-compatible"]]
```

## EmbeddingProvider

Embeddings model provider configuration type.

```python
class EmbeddingProvider(TypedDict):
    provider_name: str
    embeddings_model: EmbeddingsType
    base_url: NotRequired[str]
```

**Field Descriptions:**

- `provider_name`: string, required, provider name
- `embeddings_model`: Embeddings type or string type, required, embeddings model class or string
- `base_url`: optional string, base URL
