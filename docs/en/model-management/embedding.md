# Embedding Model Management

> [!NOTE]  
> **Feature Overview**: Provides more efficient and convenient embedding model management, supporting multiple model providers.  
> **Prerequisites**: Understanding of LangChain [Embedding Models](https://docs.langchain.com/oss/python/integrations/text_embedding/).  
> **Estimated Reading Time**: 8 minutes.

## Overview

LangChain's `init_embeddings` function only supports a limited number of embedding model providers. This library offers a more flexible embedding model management solution, particularly suitable for scenarios where you need to integrate embedding services not natively supported (such as vLLM, etc.).

Using the embedding model management functionality of this library requires two steps:

1. **Register Embedding Model Provider**

Use `register_embeddings_provider` to register an embedding model provider. Its parameters are defined as follows:

<Params  
name="provider_name"  
type="string"  
description="Name of the embedding model provider, used as an identifier for subsequent model loading"  
:required="true"  
:default="null"  
/>  
<Params  
name="embeddings_model"  
type="Embeddings | string"  
description="Embedding model, can be an Embeddings class or a string (currently supports 'openai-compatible')"  
:required="true"  
:default="null"  
/>  
<Params  
name="base_url"  
type="string"  
description="API address of the embedding model provider (optional, valid for both types of embeddings_model, but mainly used when embeddings_model is a string and is 'openai-compatible')"  
:required="false"  
:default="null"  
/>

2. **Load Embedding Model**

Use `load_embeddings` to instantiate a specific embedding model. Its parameters are defined as follows:

<Params  
name="model"  
type="string"  
description="Name of the embedding model"  
:required="true"  
:default="null"  
/>  
<Params  
name="provider"  
type="string"  
description="Name of the embedding model provider"  
:required="false"  
:default="null"  
/>

**Note**: The `load_embeddings` function can also accept any number of keyword arguments, which can be used to pass additional parameters such as `dimension`.

## Registering Embedding Model Providers

To register an embedding model provider, call `register_embeddings_provider`. The registration method varies slightly depending on the type of `embeddings_model`.

<CaseItem :step="1" content="Existing LangChain Embedding Model Class"></CaseItem>

If the embedding model provider already has a suitable LangChain integration (see [Embedding Model Integration List](https://docs.langchain.com/oss/python/integrations/text_embedding)), please pass the corresponding embedding model class directly to the `embeddings_model` parameter.

Refer to the following code for specific implementation:

```python
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_dev_utils.embeddings import register_embeddings_provider

register_embeddings_provider(
    provider_name="fake_provider",
    embeddings_model=FakeEmbeddings,
)
```

The following points supplement the above code:

- **`FakeEmbeddings` is for testing only**. In actual use, you must pass a functional `Embeddings` class.
- **`provider_name` represents the name of the model provider**, used for reference in `load_chat_model` later. The name can be customized, but should not include special characters such as ":" or "-".

Additionally, in this case, the function supports passing the `base_url` parameter, but **usually there is no need to manually set `base_url`** (as the API address is already defined within the embedding model class). Only when you need to override the default address should you explicitly pass `base_url`; the override scope is limited to attributes with field names `api_base` or `base_url` (including aliases) in the model class.

<CaseItem :step="2" content="No LangChain Embedding Model Class, but Provider Supports OpenAI-Compatible API"></CaseItem>

Similar to chat models, many embedding model providers also offer **OpenAI-compatible APIs**. When there is no existing LangChain integration but the protocol is supported, this mode can be used.

This library will use `OpenAIEmbeddings` (from `langchain-openai`) to build the embedding model instance and automatically disable context length checking (set `check_embedding_ctx_length=False`) to improve compatibility.

In this case, besides passing the `provider_name` and `chat_model` (which must be `"openai-compatible"`) parameters, you also need to pass the `base_url` parameter.

For the `base_url` parameter, it can be provided in either of the following ways:

  - **Explicit parameter passing**:

  ```python
  register_embeddings_provider(
      provider_name="vllm",
      embeddings_model="openai-compatible",
      base_url="http://localhost:8000/v1"
  )
  ```

  - **Environment variable (recommended)**:

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

::: info Tip  
The environment variable naming convention is `${PROVIDER_NAME}_API_BASE` (all uppercase, separated by underscores).  
The corresponding API Key environment variable is `${PROVIDER_NAME}_API_KEY`.  
:::

::: tip Additional Information  
vLLM can deploy embedding models and expose OpenAI-compatible interfaces, for example:

```bash
vllm serve Qwen/Qwen3-Embedding-4B \
--task embed \
--served-model-name qwen3-embedding-4b \
--host 0.0.0.0 --port 8000
```

The service address is `http://localhost:8000/v1`.  
:::

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

::: warning Note  
Both registration functions are implemented based on a global dictionary. **All registrations must be completed during the application startup phase**, and dynamic registration during runtime is prohibited to avoid multithreading issues.  
:::

## Loading Embedding Models

Use `load_embeddings` to initialize embedding model instances. The parameter rules are as follows:

- If `provider` is not passed, then `model` must be in the format of `provider_name:embeddings_name`;
- If `provider` is passed, then `model` is only `embeddings_name`.

**Example**:

```python
# Method 1
embedding = load_embeddings("vllm:qwen3-embedding-4b")

# Method 2
embedding = load_embeddings("qwen3-embedding-4b", provider="vllm")
```

### Additional Parameter Support

Any keyword arguments can be passed, for example:

```python
embedding = load_embeddings(
    "fake_provider:fake-emb",
    size=1024  # Parameter required by FakeEmbeddings
)
```

For the `"openai-compatible"` type, all parameters of `OpenAIEmbeddings` are supported.

### Compatibility with Official Providers

For providers officially supported by LangChain (such as `openai`), you can directly use `load_embeddings` without registration:

```python
model = load_embeddings("openai:text-embedding-3-large")
# or
model = load_embeddings("text-embedding-3-large", provider="openai")
```

<BestPractice>
    <p>For the use of this module, you can choose based on the following three situations:</p>
    <ol>
        <li>If all embedding model providers you integrate are supported by the official <code>init_embeddings</code>, please use the official function directly for the best compatibility.</li>
        <li>If some of the embedding model providers you integrate are not officially supported, you can use the registration and loading mechanism of this module. First use <code>register_embeddings_provider</code> to register the model provider, then use <code>load_embeddings</code> to load the model.</li>
        <li>If the embedding model provider you integrate does not have a suitable integration, but the provider offers an OpenAI-compatible API (such as vLLM, OpenRouter), it is recommended to use the functionality of this module. First use <code>register_embeddings_provider</code> to register the model provider (passing <code>openai-compatible</code> for embeddings_model), then use <code>load_embeddings</code> to load the model.</li>
    </ol>
</BestPractice>