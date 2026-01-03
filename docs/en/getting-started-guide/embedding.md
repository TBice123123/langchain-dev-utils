# Embedding Model Management

## Overview

LangChain's `init_embeddings` function only supports a limited number of embedding model providers. This library provides a more flexible embedding model management solution, particularly suitable for scenarios where you need to integrate embedding services not natively supported (such as vLLM).

## Registering Embedding Model Providers

To register an embedding model provider, call `register_embeddings_provider`. The registration method varies slightly depending on the type of `embeddings_model`.

### Existing LangChain Embedding Model Class

If the embedding model provider already has a ready and suitable LangChain integration (see [Embedding Model Integration List](https://docs.langchain.com/oss/python/integrations/text_embedding)), directly pass the corresponding embedding model class to the `embeddings_model` parameter.

#### Parameter Description

| Parameter | Description |
|-----------|-------------|
| `provider_name` | Model provider name, used for subsequent reference in `load_embeddings`.<br><br>**Type**: `str`<br>**Required**: Yes |
| `embeddings_model` | LangChain embedding model class.<br><br>**Type**: `type[Embeddings]`<br>**Required**: Yes |
| `base_url` | API base URL, usually no need to set manually.<br><br>**Type**: `str`<br>**Required**: No |

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
- `provider_name` represents the name of the model provider, used for subsequent reference in `load_embeddings`. The name can be customized, but should not contain special characters such as `:`, `-`, etc.

!!! warning "Warning"
    `provider_name` must start with a letter or number, can only contain letters, numbers, and underscores, and must be 20 characters or fewer.

#### Optional Parameters

**base_url**

This parameter usually does not need to be set (because the embedding model class already defines a default API address internally). Only pass `base_url` when you need to override the default address of the embedding model class, and it only takes effect for fields named `api_base` or `base_url` (including aliases).


### No LangChain Embedding Model Class, but Provider Supports OpenAI Compatible API

The parameter description for this situation is as follows:

#### Parameter Description

| Parameter | Description |
|-----------|-------------|
| `provider_name` | Model provider name, used for subsequent reference in `load_embeddings`.<br><br>**Type**: `str`<br>**Required**: Yes |
| `embeddings_model` | Fixed value `"openai-compatible"`.<br><br>**Type**: `str`<br>**Required**: Yes |
| `base_url` | API base URL.<br><br>**Type**: `str`<br>**Required**: No |

#### Code Example

**Method 1: Explicit Parameters**

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

**Note**: For more details on this part, please refer to [OpenAI Compatible API Integration](../adavance-guide/openai-compatible.md).

## Batch Registration

If you need to register multiple providers, you can use `batch_register_embeddings_provider`.

#### Parameter Description

| Parameter | Description |
|-----------|-------------|
| `providers` | List of provider configurations, each dictionary contains registration parameters.<br><br>**Type**: `list[dict]`<br>**Required**: Yes |

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
    
    Additionally, when registering with `embeddings_model` set to `openai-compatible`, the system internally uses `pydantic.create_model` to dynamically create new model classes (with `BaseEmbeddingOpenAICompatible` as the base class, generating corresponding embedding model integration classes). This process involves Python metaclass operations and pydantic validation logic initialization, which has certain performance overhead, so please avoid frequent registration during runtime.

## Loading Embedding Models

Use `load_embeddings` to initialize embedding model instances.

#### Parameter Description

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `str` | Yes | - | Model name |
| `provider` | `str` | No | `None` | Model provider name |

**In addition, any number of keyword arguments can be passed to provide additional parameters for the embedding model class.**

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

For supported model methods and parameters, refer to the usage instructions of the corresponding embedding model class. If you're using the second situation, all methods and parameters of the `OpenAIEmbeddings` class are supported.

### Compatibility with Official Providers

For providers already supported by LangChain (such as `openai`), you can directly use `load_embeddings` without registration:

```python
model = load_embeddings("openai:text-embedding-3-large")
# or
model = load_embeddings("text-embedding-3-large", provider="openai")
```

!!! success "Best Practice"
    For the use of this module, you can choose based on the following three situations:

    1. If all embedding model providers you're integrating are supported by the official `init_embeddings`, please use the official function directly for the best compatibility.

    2. If some embedding model providers you're integrating are not officially supported, you can use the registration and loading mechanism of this module. First, use `register_embeddings_provider` to register the model provider, then use `load_embeddings` to load the model.

    3. If the embedding model provider you're integrating doesn't have a suitable integration yet, but the provider offers an OpenAI-compatible API (such as vLLM), it's recommended to use the functionality of this module. First, use `register_embeddings_provider` to register the model provider (passing `openai-compatible` for embeddings_model), then use `load_embeddings` to load the model.