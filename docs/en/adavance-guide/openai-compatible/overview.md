# Overview

!!! warning "Prerequisites"
    To use this feature, you must install the standard version of the `langchain-dev-utils` library. Please refer to the installation section for details.

Many model providers offer **OpenAI Compatible API** services, such as [vLLM](https://github.com/vllm-project/vllm), [OpenRouter](https://openrouter.ai/), and [Together AI](https://www.together.ai/). This library provides an OpenAI Compatible API integration solution covering both chat models and embedding models. It is especially suitable for scenarios where "a provider offers an OpenAI Compatible API but lacks corresponding LangChain integration."

This library provides two utility functions for creating chat model and embedding model integration classes:

| Function Name | Description |
|--------|------|
| `create_openai_compatible_model` | Creates a chat model integration class |
| `create_openai_compatible_embedding` | Creates an embedding model integration class |

!!! tip "Note"
    The initial inspiration for the two utility functions provided by this library came from the JavaScript ecosystem's [@ai-sdk/openai-compatible](https://ai-sdk.dev/providers/openai-compatible-providers).

This documentation will use integrating [vLLM](https://github.com/vllm-project/vllm) as an example to demonstrate how to use this feature.

??? note "vLLM Introduction"
    vLLM is a commonly used large model inference framework, ideal for high-performance inference services in local or self-hosted environments. It can deploy large models as OpenAI compatible APIs, facilitating the reuse of existing SDKs and calling methods. It supports deployment for both chat models and embedding models, as well as capabilities like multi-model serving, tool calling, and inference output, making it suitable for scenarios such as dialogue, tool calling, and multimodal tasks.

    The following examples are model deployment commands that will be used later in the content:

    **Qwen3-4B**:

    ```bash
    vllm serve Qwen/Qwen3-4B \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3-4b
    ```

    **GLM-4.7-Flash**:

    ```bash
    vllm serve zai-org/GLM-4.7-Flash \
     --tensor-parallel-size 4 \
     --speculative-config.method mtp \
     --speculative-config.num_speculative_tokens 1 \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --served-model-name glm-4.7-flash
    ```

    **Qwen3-VL-2B-Instruct**:

    ```bash
    vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3-vl-2b
    ```

    **Qwen3-Embedding-4B**:

    ```bash
    vllm serve Qwen/Qwen3-Embedding-4B \
    --task embed \
    --served-model-name qwen3-embedding-4b \
    --host 0.0.0.0 --port 8000
    ```
    The service address is `http://localhost:8000/v1`.