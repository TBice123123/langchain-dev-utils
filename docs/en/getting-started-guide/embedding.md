# Embedding Model Management

## Overview

LangChain's `init_embeddings` function only supports a limited number of embedding model providers. This library provides a more flexible embedding model management solution, especially suitable for scenarios where you need to integrate embedding services not natively supported (such as vLLM, etc.).

## Registering Embedding Model Providers

To register an embedding model provider, you need to call `register_embeddings_provider`. The registration method varies slightly depending on the `embeddings_model` type.

### Existing LangChain Embedding Model Class

If the embedding model provider already has a suitable LangChain integration (see [Embedding Model Integration List](https://docs.langchain.com/oss/python/integrations/text_embedding)), please pass the corresponding embedding model class directly to the `embeddings_model` parameter.

For specific code, refer to the following example:

```python
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_dev_utils.embeddings import register_embeddings_provider

register_embeddings_provider(
    provider_name="fake_provider",
    embeddings_model=FakeEmbeddings,
)
```

Additional notes for the above code:

- `FakeEmbeddings` is only for testing. In actual use, you must pass an `Embeddings` class with real functionality.
- `provider_name` represents the name of the model provider, used for later reference in `load_chat_model`. The name can be customized but should not contain special characters such as ":" or "-".

At the same time, in this case, the function also supports passing the `base_url` parameter, but **usually you don't need to manually set `base_url`** (because the embedding model class already defines the API address internally). Only when you need to override the default address should you explicitly pass `base_url`; the override scope is limited to attributes with field names `api_base` or `base_url` (including aliases) in the model class.

### No LangChain Embedding Model Class, but Provider Supports OpenAI Compatible API

Similar to chat models, many embedding model providers also provide **OpenAI Compatible API**. When there is no ready-made LangChain integration but the protocol is supported, you can use this mode.

This library will use `OpenAIEmbeddings` (from `langchain-openai`) to build an embedding model instance and automatically disable context length checking (set `check_embedding_ctx_length=False`) to improve compatibility.

In this case, in addition to passing `provider_name` and `embeddings_model` (which must be `"openai-compatible"`), you also need to pass the `base_url` parameter.

For the `base_url` parameter, you can provide it in either of the following ways:

  - **Explicit parameter passing**:

  ```python
  register_embeddings_provider(
      provider_name="vllm",
      embeddings_model="openai-compatible",
      base_url="http://localhost:8000/v1"
  )
  ```

  - **Environment Variable (Recommended)**:

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

!!! info "Tip"  
    The environment variable naming rule is `${PROVIDER_NAME}_API_BASE` (all uppercase, separated by underscores).  
    The corresponding API Key environment variable is `${PROVIDER_NAME}_API_KEY`.


!!! note "Additional"  
    vLLM can deploy embedding models and expose OpenAI compatible interfaces, for example:

    ```bash
    vllm serve Qwen/Qwen3-Embedding-4B \
    --task embed \
    --served-model-name qwen3-embedding-4b \
    --host 0.0.0.0 --port 8000
    ```

    The service address is `http://localhost:8000/v1`.


## Batch Registration

If you need to register multiple providers, you can use `batch_register_embeddings_provider`:

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

Use `load_embeddings` to initialize an embedding model instance. The parameter rules are as follows:

- If `provider` is not passed, then `model` must be in the format `provider_name:embeddings_name`;
- If `provider` is passed, then `model` is only `embeddings_name`.

**Example**:

```python
# Method 1
embedding = load_embeddings("vllm:qwen3-embedding-4b")

# Method 2
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

For providers already officially supported by LangChain (such as `openai`), you can directly use `load_embeddings` without registration:

```python
model = load_embeddings("openai:text-embedding-3-large")
# or
model = load_embeddings("text-embedding-3-large", provider="openai")
```

!!! success "Best Practice"
    For the use of this module, you can choose according to the following three situations:

    1. If all embedding model providers you're integrating are supported by the official `init_embeddings`, please use the official function directly to get the best compatibility.

    2. If some of the embedding model providers you're integrating are not officially supported, you can use the registration and loading mechanism of this module, first use `register_embeddings_provider` to register the model provider, then use `load_embeddings` to load the model.

    3. If the embedding model provider you're integrating does not have a suitable integration yet, but the provider provides an OpenAI compatible API (such as vLLM), it's recommended to use the functionality of this module, first use `register_embeddings_provider` to register the model provider (pass `openai-compatible` for embeddings_model), then use `load_embeddings` to load the model.