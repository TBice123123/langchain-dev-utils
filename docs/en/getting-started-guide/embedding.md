# Embedding Model Management

## Overview

LangChain's `init_embeddings` function only supports a limited number of embedding model providers. This library provides a more flexible embedding model management solution, particularly suitable for scenarios where you need to connect to embedding services not natively supported (such as vLLM, etc.).

## Registering Embedding Model Providers

To register an embedding model provider, call `register_embeddings_provider`. The registration method varies slightly depending on the type of `embeddings_model`.

### Existing LangChain Embedding Model Classes

If an embedding model provider already has a ready and suitable LangChain integration (see [Embedding Model Integration List](https://docs.langchain.com/oss/python/integrations/text_embedding)), please pass the corresponding embedding model class directly to the `embeddings_model` parameter.

Refer to the following code for specific implementation:

```python
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_dev_utils.embeddings import register_embeddings_provider

register_embeddings_provider(
    provider_name="fake_provider",
    embeddings_model=FakeEmbeddings,
)
```

Additional notes regarding the above code:

- `FakeEmbeddings` is for testing purposes only. In actual use, you must pass a functional `Embeddings` class.
- `provider_name` represents the name of the model provider, used for subsequent reference in `load_chat_model`. The name can be customized but should not include special characters such as ":" or "-".

Additionally, in this case, the function also supports passing the `base_url` parameter, but **you usually don't need to manually set `base_url`** (since the API address is already defined within the embedding model class). Only pass `base_url` explicitly when you need to override the default address; the override scope is limited to attributes with field names `api_base` or `base_url` (including aliases) in the model class.

### No LangChain Embedding Model Class Available, but Provider Supports OpenAI Compatible API

Similar to chat models, many embedding model providers also offer **OpenAI Compatible APIs**. When there's no ready LangChain integration but the protocol is supported, you can use this mode.

This library will use `OpenAIEmbeddings` (from `langchain-openai`) to build the embedding model instance and automatically disable context length checking (setting `check_embedding_ctx_length=False`) to improve compatibility.

In this case, besides passing the `provider_name` and `chat_model` (which must be set to `"openai-compatible"`) parameters, you also need to pass the `base_url` parameter.

For the `base_url` parameter, it can be provided through either of the following methods:

  - **Explicit parameter passing**:

  ```python
  register_embeddings_provider(
      provider_name="vllm",
      embeddings_model="openai-compatible",
      base_url="http://localhost:8000/v1"
  )
  ```

  - **Environment variables (recommended)**:

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
    The environment variable naming convention is `${PROVIDER_NAME}_API_BASE` (all uppercase, separated by underscores).  
    The corresponding API Key environment variable is `${PROVIDER_NAME}_API_KEY`.


!!! note "Additional Information"  
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
    Both registration functions are implemented based on a global dictionary. **All registrations must be completed during the application startup phase**. Dynamic registration during runtime is prohibited to avoid multithreading issues.
    

## Loading Embedding Models

Use `load_embeddings` to initialize embedding model instances. The parameter rules are as follows:

- If `provider` is not passed, then `model` must be in the format `provider_name:embeddings_name`;
- If `provider` is passed, then `model` is only `embeddings_name`.

**Examples**:

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
    For using this module, you can choose based on the following three scenarios:

    1. If all embedding model providers you're connecting to are supported by the official `init_embeddings`, please use the official function directly for best compatibility.

    2. If some of the embedding model providers you're connecting to are not officially supported, you can use the registration and loading mechanism of this module. First, register the model provider using `register_embeddings_provider`, then load the model using `load_embeddings`.
    
    3. If the embedding model provider you're connecting to doesn't have a suitable integration yet, but the provider offers an OpenAI compatible API (such as vLLM, OpenRouter), it's recommended to use this module's functionality. First, register the model provider using `register_embeddings_provider` (passing `openai-compatible` to `embeddings_model`), then load the model using `load_embeddings`.