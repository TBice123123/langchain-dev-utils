# Embedding Model Management

## Overview

LangChain's `init_embeddings` function only supports a limited number of embedding model providers. This library provides a more flexible embedding model management solution, especially suitable for scenarios where you need to integrate embedding services not natively supported (such as vLLM).

## Registering Embedding Model Providers

To register an embedding model provider, you need to call `register_embeddings_provider`. The registration method varies slightly depending on the type of `embeddings_model`.

### Existing LangChain Embedding Model Class

If the embedding model provider already has a ready and suitable LangChain integration (see [Embedding Model Integration List](https://docs.langchain.com/oss/python/integrations/text_embedding)), please pass the corresponding embedding model class directly to the `embeddings_model` parameter.

#### Parameter Description

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `provider_name` | `str` | Yes | - | Model provider name, used for reference in `load_embeddings` |
| `embeddings_model` | `type[Embeddings]` | Yes | - | LangChain embedding model class |
| `base_url` | `str` | No | `None` | API base address, usually no need to set manually |

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
- `provider_name` represents the name of the model provider, used for reference in `load_embeddings`. The name can be customized, but should not contain special characters like `:`, `-`, etc.
- The `base_url` parameter usually does not need to be set manually (since the embedding model class already defines the API address internally). Only when you need to override the default address should you explicitly pass `base_url`; the override scope is limited to attributes with field names `api_base` or `base_url` (including aliases) in the model class.

### No LangChain Embedding Model Class, but Provider Supports OpenAI Compatible API

The parameter description for this situation is as follows:

#### Parameter Description

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `provider_name` | `str` | Yes | - | Model provider name |
| `embeddings_model` | `str` | Yes | - | Fixed value `"openai-compatible"` |
| `base_url` | `str` | No | `None` | API base address |

#### Code Example

**Method 1: Explicit Parameter Passing**

```python
register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible",
    base_url="http://localhost:8000/v1"
)
```

**Method 2: Environment Variables (Recommended)**

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

**Note**: For more details on this part, please refer to the [OpenAI Compatible API Integration](../adavance-guide/openai-compatible.md).


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
    Both registration functions are implemented based on global dictionaries. **All registrations must be completed during the application startup phase**, and dynamic registration during runtime is prohibited to avoid multithreading issues.

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

## Model Methods and Parameters

For supported model methods and parameters, please refer to the usage instructions of the corresponding embedding model class. If you are using the second situation, then all methods and parameters of the `OpenAIEmbeddings` class are supported.

### Compatibility with Official Providers

For providers already officially supported by LangChain (such as `openai`), you can directly use `load_embeddings` without registration:

```python
model = load_embeddings("openai:text-embedding-3-large")
# or
model = load_embeddings("text-embedding-3-large", provider="openai")
```

!!! success "Best Practice"
    For the use of this module, you can choose according to the following three situations:

    1. If all embedding model providers you integrate are already supported by the official `init_embeddings`, please use the official function directly to get the best compatibility.

    2. If some embedding model providers you integrate are not officially supported, you can use the registration and loading mechanism of this module, first register model providers using `register_embeddings_provider`, then use `load_embeddings` to load models.

    3. If the embedding model provider you integrate does not have a suitable integration yet, but the provider provides an OpenAI compatible API (such as vLLM), it is recommended to use the functionality of this module, first register the model provider using `register_embeddings_provider` (pass `openai-compatible` to embeddings_model), then use `load_embeddings` to load the model.