# Creation and Usage of Embedding Models

## Creating Embedding Model Classes

Similar to chat model classes, you can use `create_openai_compatible_embedding` to create an integrated embedding model class. This function accepts the following parameters:

| Parameter | Description |
|-----------|-------------|
| `embedding_provider` | Embedding model provider name, e.g., `vllm`. Must start with a letter or number, can only contain letters, numbers, and underscores, with a maximum length of 20 characters.<br><br>**Type**: `str`<br>**Required**: Yes |
| `base_url` | Default API endpoint for the model provider.<br><br>**Type**: `str`<br>**Required**: No |
| `embedding_model_cls_name` | Embedding model class name (must comply with Python class naming conventions). Default value is `{Provider}Embeddings` (where `{Provider}` is the capitalized provider name).<br><br>**Type**: `str`<br>**Required**: No |

Similarly, we use `create_openai_compatible_embedding` to integrate vLLM's embedding model.

```python hl_lines="4 5 6"
from langchain_dev_utils.embeddings.adapters import create_openai_compatible_embedding

VLLMEmbeddings = create_openai_compatible_embedding(
    embedding_provider="vllm",
    base_url="http://localhost:8000/v1",
    embedding_model_cls_name="VLLMEmbeddings",
)

embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_query("Hello"))
```

`base_url` can also be omitted. If not provided, the library will read the environment variable `VLLM_API_BASE` by default:

```bash
export VLLM_API_BASE="http://localhost:8000/v1"
```

At this point, the code can omit `base_url`:

```python hl_lines="4 5"
from langchain_dev_utils.embeddings.adapters import create_openai_compatible_embedding

VLLMEmbeddings = create_openai_compatible_embedding(
    embedding_provider="vllm",
    embedding_model_cls_name="VLLMEmbeddings",
)

embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_query("Hello"))
```

**Note**: The above code successfully runs assuming the environment variable `VLLM_API_KEY` is configured. Although vLLM itself does not require an API Key, the embedding model class initialization requires one. Therefore, please set this variable first, for example:

```bash
export VLLM_API_KEY=vllm_api_key
```

## Using the Embedding Model Class

Here, we use the previously created `VLLMEmbeddings` class to initialize an embedding model instance.

### Vectorizing Queries

```python
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_query("Hello"))
```

Similarly, asynchronous invocation is also supported:

```python
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
res = await embedding.aembed_query("Hello")
print(res)
```

### Vectorizing a List of Strings

```python
documents = ["Hello", "Hello, I am Zhang San"]
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_documents(documents))
```
Similarly, asynchronous invocation is also supported:

```python
documents = ["Hello", "Hello, I am Zhang San"]
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
res = await embedding.aembed_documents(documents)
print(res)
```

!!! warning "Embedding Model Compatibility Notes"
    OpenAI-compatible embedding APIs generally exhibit good compatibility, but the following differences should be noted:

    1. `check_embedding_ctx_length`: Set to `True` only when using the official OpenAI embedding service; for all other embedding models, set it to `False`.

    2. `dimensions`: If the model supports custom vector dimensions (e.g., 1024, 4096), you can directly pass this parameter.

    3. `chunk_size`: The maximum number of texts that can be processed in a single API call. For example, a `chunk_size` of 10 means a single request can vectorize up to 10 texts.
    
    4. Single-text token limit: Cannot be controlled via parameters; must be ensured during preprocessing and chunking stages.

!!! warning "Note"
    Similarly, this function uses [pydantic's create_model](https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation) under the hood to create the embedding model class, which incurs a certain performance overhead. It is recommended to create the integration class during the project startup phase and avoid dynamic creation later on.


!!! success "Best Practice"
    When connecting to an OpenAI-compatible API embedding model provider, you can directly use `langchain-openai`'s `OpenAIEmbeddings` and point `base_url` and `api_key` to your provider's service. Embedding model API compatibility is usually better: in most cases, you can directly use `OpenAIEmbeddings` with `check_embedding_ctx_length=False`.
    