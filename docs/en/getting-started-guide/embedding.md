# Embedding Model Management

## Overview

LangChain's `init_embeddings` function only supports a limited number of embedding model providers. This library provides a more flexible embedding model management solution, particularly suitable for scenarios where you need to integrate embedding services not natively supported (such as vLLM, etc.).

## Registering Embedding Model Providers

To register an embedding model provider, you need to call `register_embeddings_provider`. The registration method varies slightly depending on the type of `embeddings_model`.

### Existing LangChain Embedding Model Class

If the embedding model provider already has a suitable and ready-to-use LangChain integration (see [Embedding Model Integration List](https://docs.langchain.com/oss/python/integrations/text_embedding)), please directly pass the corresponding embedding model class to the `embeddings_model` parameter.

#### Parameter Description

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `provider_name` | `str` | Yes | - | Model provider name, used for reference in `load_embeddings` later |
| `embeddings_model` | `type[Embeddings]` | Yes | - | LangChain embedding model class |
| `base_url` | `str` | No | `None` | API base URL, usually no need to set manually |

#### Code Example

```python
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_dev_utils.embeddings import register_embeddings_provider

register_embeddings_provider(
    provider_name="fake_provider",
    embeddings_model=FakeEmbeddings,
)
```

#### Usage Instructions

- `FakeEmbeddings` is only for testing. In actual use, you must pass an `Embeddings` class with real functionality.
- `provider_name` represents the name of the model provider, used for reference in `load_embeddings` later. The name can be customized, but should not contain special characters such as `:`, `-`, etc.
- The `base_url` parameter usually doesn't need to be set manually (since the embedding model class already defines the API address internally). Only when you need to override the default address, should you explicitly pass `base_url`; the override scope is limited to attributes with field names `api_base` or `base_url` (including aliases) in the model class.

### No LangChain Embedding Model Class, but Provider Supports OpenAI Compatible API

Similar to chat models, many embedding model providers also offer **OpenAI Compatible API**. When there is no ready-to-use LangChain integration but the protocol is supported, you can use this mode.

This library will use `OpenAIEmbeddings` (from `langchain-openai`) to build embedding model instances, and automatically disable context length checking (set `check_embedding_ctx_length=False`) to improve compatibility.

#### Parameter Description

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `provider_name` | `str` | Yes | - | Model provider name |
| `embeddings_model` | `str` | Yes | - | Fixed value `"openai-compatible"` |
| `base_url` | `str` | No | `None` | API base URL |

#### Code Example

**Method 1: Explicit parameters**

```python
register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible",
    base_url="http://localhost:8000/v1"
)
```

**Method 2: Environment variables (recommended)**

```bash
export VLLM_API_BASE=http://localhost:8000/v1
```

```python
register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible"
    # Automatically reads VLLM_API_BASE
)
```

#### Environment Variable Description

| Environment Variable | Description |
|----------------------|-------------|
| `${PROVIDER_NAME}_API_BASE` | API base URL (all caps, underscore separated) |
| `${PROVIDER_NAME}_API_KEY` | API key |

!!! info "Tip"
    The naming rule for environment variables is `${PROVIDER_NAME}_API_BASE` (all caps, underscore separated).
    The corresponding API Key environment variable is `${PROVIDER_NAME}_API_KEY`.

!!! note "Supplement"  
    vLLM can deploy embedding models and expose OpenAI compatible interfaces, for example:

    ```bash
    vllm serve Qwen/Qwen3-Embedding-4B \
    --task embed \
    --served-model-name qwen3-embedding-4b \
    --host 0.0.0.0 --port 8000
    ```

    The service address is `http://localhost:8000/v1`.

## Batch Registration

If you need to register multiple providers, you can use `batch_register_embeddings_provider`.

#### Parameter Description

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `providers` | `list[dict]` | Yes | - | List of provider configurations, each dictionary contains registration parameters |

#### Code Example

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

!!! warning "Note"
    Both registration functions are implemented based on a global dictionary. **All registrations must be completed during the application startup phase**, and dynamic registration during runtime is prohibited to avoid multi-threading issues.

## Loading Embedding Models

Use `load_embeddings` to initialize embedding model instances.

#### Parameter Description

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `str` | Yes | - | Model name |
| `provider` | `str` | No | `None` | Model provider name |

**In addition, you can pass any number of keyword arguments for additional parameters of the embedding model class.**

#### Parameter Rules

- If `provider` is not passed, then `model` must be in the format `provider_name:embeddings_name`;
- If `provider` is passed, then `model` is only `embeddings_name`.

#### Code Example

```python
# Method 1: model includes provider information
embedding = load_embeddings("vllm:qwen3-embedding-4b")

# Method 2: specify provider separately
embedding = load_embeddings("qwen3-embedding-4b", provider="vllm")
```

### Additional Parameter Support

You can pass any keyword arguments, for example:

```python
embedding = load_embeddings(
    "fake_provider:fake-emb",
    size=1024  # Parameter required by FakeEmbeddings
)
```

For the `"openai-compatible"` type, all parameters of `OpenAIEmbeddings` are supported.

### Compatibility with Official Providers

For providers already supported by LangChain official (such as `openai`), you can directly use `load_embeddings` without registration:

```python
model = load_embeddings("openai:text-embedding-3-large")
# or
model = load_embeddings("text-embedding-3-large", provider="openai")
```

!!! success "Best Practice"
    For the use of this module, you can choose according to the following three situations:

    1. If all embedding model providers you're integrating are supported by the official `init_embeddings`, please directly use the official function to get the best compatibility.

    2. If some of the embedding model providers you're integrating are not officially supported, you can use the registration and loading mechanism of this module, first register the model provider using `register_embeddings_provider`, then use `load_embeddings` to load the model.

    3. If the embedding model provider you're integrating does not have a suitable integration, but the provider provides an OpenAI Compatible API (such as vLLM), it is recommended to use the functionality of this module, first register the model provider using `register_embeddings_provider` (pass `openai-compatible` as embeddings_model), then use `load_embeddings` to load the model.