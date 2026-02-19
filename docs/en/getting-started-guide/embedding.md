# Embedding Model Management

## Overview

LangChain's `init_embeddings` function only supports a limited number of embedding model providers. This library provides a more flexible embedding model management solution, particularly suitable for scenarios requiring integration with embedding services not natively supported (such as vLLM, etc.).

## Registering Embedding Model Providers

To register an embedding model provider, call `register_embeddings_provider`. The registration method varies slightly depending on the `embeddings_model` type.

### Existing LangChain Embedding Model Class

If the embedding model provider already has a ready-made and suitable LangChain integration (see [Embedding Model Integration List](https://docs.langchain.com/oss/python/integrations/text_embedding)), pass the corresponding embedding model class directly as the `embeddings_model` parameter.

#### Code Example

```python hl_lines="5 6"
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_dev_utils.embeddings import register_embeddings_provider

register_embeddings_provider(
    provider_name="fake_provider",
    embeddings_model=FakeEmbeddings,
)

# FakeEmbeddings is for testing only; in actual usage, you must pass an Embeddings class with real functionality.
```

!!! tip "Parameter Setting Instructions"
    `provider_name` represents the name of the model provider, used for subsequent reference in `load_embeddings`. The name must start with a letter or number, contain only letters, numbers, and underscores, and be no longer than 20 characters.

#### Optional Parameter Description

**base_url**

This parameter usually does not need to be set (since embedding model classes generally define a default API address internally). Only pass `base_url` when you need to override the default address of the embedding model class, and it only takes effect for attributes named `api_base` or `base_url` (including aliases).

### No LangChain Embedding Model Class, but Provider Supports OpenAI-Compatible API

Similar to chat model management, set `embeddings_model` to `"openai-compatible"`.

#### Code Example

```python hl_lines="2 3 4"
register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible",
    base_url="http://localhost:8000/v1"
)
```

**Note**: For more details on this part, please refer to [OpenAI-Compatible API Integration](../adavance-guide/openai-compatible/register.md).

## Batch Registration

If you need to register multiple providers, you can use `batch_register_embeddings_provider`.

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
    Both registration functions are implemented based on a global dictionary. To avoid multi-threading issues, **all registrations must be completed during application startup**; dynamic registration at runtime is prohibited.

    Additionally, if `embeddings_model` is set to `openai-compatible` during registration, a new model class is dynamically created internally via `pydantic.create_model` (generating the corresponding embedding model integration class based on `BaseEmbeddingOpenAICompatible`). This process involves Python metaclass operations and pydantic validation logic initialization, which incurs some performance overhead. Therefore, please avoid frequent registration during runtime.

## Loading Embedding Models

Use `load_embeddings` to initialize an embedding model instance.

This function accepts a `model` parameter to specify the model name, an optional `provider` parameter to specify the model provider, and any number of keyword arguments to pass extra parameters to the embedding model class.

#### Parameter Rules

- If `provider` is not passed, `model` must be in the `provider_name:embeddings_name` format;
- If `provider` is passed, `model` must be just `embeddings_name`.

#### Code Example

```python
# Method 1: model includes provider info
embedding = load_embeddings("vllm:qwen3-embedding-4b")

# Method 2: Specify provider separately
embedding = load_embeddings("qwen3-embedding-4b", provider="vllm")
```

## Model Methods and Parameters

For supported model methods and parameters, refer to the usage documentation of the corresponding embedding model class. If using the second case, all methods and parameters of the `OpenAIEmbeddings` class are supported.

### Compatibility with Official Providers

`load_embeddings` looks up the global registration dictionary based on the `provider` parameter: if found, it instantiates the model class from that dictionary; if not found, it initializes via `init_embeddings`. This means providers officially supported by LangChain (such as openai) can be called directly without registration.

```python
model = load_embeddings("openai:text-embedding-3-large")
# Or
model = load_embeddings("text-embedding-3-large", provider="openai")
```

!!! success "Best Practices"
    For the usage of this module, you can choose based on the following three situations:

    1. If all embedding model providers being integrated are supported by the official `init_embeddings`, please use the official function directly for optimal compatibility.

    2. If some embedding model providers being integrated are not officially supported, you can use this module's registration and loading mechanism: first register the model provider using `register_embeddings_provider`, and then load the model using `load_embeddings`.

    3. If there is no suitable integration for the embedding model provider yet, but the provider offers an OpenAI-compatible API (such as vLLM), it is recommended to use this module's functionality: first register the model provider using `register_embeddings_provider` (pass `openai-compatible` to `embeddings_model`), and then load the model using `load_embeddings`.
