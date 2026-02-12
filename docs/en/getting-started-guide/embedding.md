# Embedding Model Management

## Overview

LangChain's `init_embeddings` function only supports a limited number of embedding model providers. This library provides a more flexible embedding model management solution, particularly suitable for scenarios requiring integration with embedding services not natively supported (such as vLLM, etc.).

## Registering Embedding Model Providers

Registering an embedding model provider requires calling `register_embeddings_provider`. The registration method varies slightly depending on the type of `embeddings_model`.

### Existing LangChain Embedding Model Classes

If the embedding model provider already has a ready-made and suitable LangChain integration (see [Embedding Model Integration List](https://docs.langchain.com/oss/python/integrations/text_embedding)), pass the corresponding embedding model class directly as the `embeddings_model` parameter.

#### Code Example

```python hl_lines="5 6"
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_dev_utils.embeddings import register_embeddings_provider

register_embeddings_provider(
    provider_name="fake_provider",
    embeddings_model=FakeEmbeddings,
)

# FakeEmbeddings is for testing only. In actual usage, you must pass an Embeddings class with real functionality.
```

!!! tip "Parameter Setting Instructions"
    `provider_name` represents the name of the model provider, used for subsequent referencing in `load_embeddings`. `provider_name` must start with a letter or number, can only contain letters, numbers, and underscores, with a maximum length of 20 characters.

#### Optional Parameter Description

**base_url**

This parameter typically does not need to be set (because embedding model classes usually define a default API address internally). Pass `base_url` only when you need to override the embedding model class's default address, and it only affects attributes with field names `api_base` or `base_url` (including aliases).

### No Existing LangChain Embedding Model Class, but Provider Supports OpenAI-Compatible API

Similar to chat model management, set `embeddings_model` to `"openai-compatible"`.

#### Code Example

```python hl_lines="2 3 4"
register_embeddings_provider(
    provider_name="vllm",
    embeddings_model="openai-compatible",
    base_url="http://localhost:8000/v1"
)
```

**Note**: For more details about this part, please refer to [OpenAI-Compatible API Integration](../adavance-guide/openai-compatible/register.md).

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
    Both registration functions are implemented based on a global dictionary. **All registrations must be completed during application startup**; dynamic registration during runtime is prohibited to avoid multi-threading issues.

    Additionally, when setting `embeddings_model` to `openai-compatible` during registration, the library dynamically creates a new model class internally using `pydantic.create_model` (generating the corresponding embedding model integration class based on `BaseEmbeddingOpenAICompatible`). This process involves Python metaclass operations and pydantic validation logic initialization, incurring some performance overhead. Therefore, avoid frequent registration during runtime.

## Loading Embedding Models

Use `load_embeddings` to initialize an embedding model instance.

This function receives the `model` parameter to specify the model name, the optional `provider` parameter to specify the model provider, and can also accept any number of keyword arguments for passing additional parameters to the embedding model class.

#### Parameter Rules

- If `provider` is not passed, `model` must be in the format `provider_name:embeddings_name`;
- If `provider` is passed, `model` must be only `embeddings_name`.

#### Code Example

```python
# Method 1: model includes provider information
embedding = load_embeddings("vllm:qwen3-embedding-4b")

# Method 2: separately specify provider
embedding = load_embeddings("qwen3-embedding-4b", provider="vllm")
```

## Model Methods and Parameters

For supported model methods and parameters, refer to the usage documentation of the corresponding embedding model class. If you are using the second case, all methods and parameters of the `OpenAIEmbeddings` class are supported.

### Compatibility with Official Providers

`load_embeddings` looks up the global registration dictionary based on the `provider` parameter: if found, it instantiates using the corresponding model class from the dictionary; if not found, it initializes via `init_embeddings`. This means providers officially supported by LangChain (such as OpenAI) can be called directly without registration.


```python
model = load_embeddings("openai:text-embedding-3-large")
# or
model = load_embeddings("text-embedding-3-large", provider="openai")
```

!!! success "Best Practice"
    For using this module, you can choose based on the following three situations:

    1. If all embedding model providers you are integrating are supported by the official `init_embeddings`, use the official function directly for the best compatibility.

    2. If some embedding model providers you are integrating are not officially supported, utilize this module's registration and loading mechanism: first register the model provider using `register_embeddings_provider`, then load the model using `load_embeddings`.

    3. If the embedding model provider you are integrating does not have a suitable integration yet, but the provider offers an OpenAI-compatible API (such as vLLM), it is recommended to use this module's functionality: first register the model provider using `register_embeddings_provider` (passing `openai-compatible` for `embeddings_model`), then load the model using `load_embeddings`.