# 概述

!!! warning "前提条件"
    使用此功能时，必须安装 standard 版本的 `langchain-dev-utils` 库。具体可以参考安装部分的介绍。


许多模型提供商都提供 **OpenAI 兼容 API** 服务，例如 [vLLM](https://github.com/vllm-project/vllm)、[OpenRouter](https://openrouter.ai/) 和 [Together AI](https://www.together.ai/) 等。本库提供一套 OpenAI 兼容 API 集成方案，覆盖对话模型与嵌入模型，尤其适用于「提供商已提供 OpenAI 兼容 API，但尚无对应 LangChain 集成」的场景。

本库提供了两个工具函数，用于创建对话模型集成类与嵌入模型集成类：

| 函数名 | 说明 |
|--------|------|
| `create_openai_compatible_model` | 创建对话模型集成类 |
| `create_openai_compatible_embedding` | 创建嵌入模型集成类 |


!!! tip "说明"
    本库提供的两个工具函数的最初灵感借鉴自 JavaScript 生态的 [@ai-sdk/openai-compatible](https://ai-sdk.dev/providers/openai-compatible-providers)。

本文档将以接入 [vLLM](https://github.com/vllm-project/vllm) 为例，展示如何使用本功能。

??? note "vLLM 介绍"
    vLLM 是常用的大模型推理框架，适合本地或自建环境下的高性能推理服务。它可以将大模型部署为 OpenAI 兼容的 API，便于复用现有的 SDK 与调用方式；同时支持对话模型与嵌入模型的部署，以及多模型服务、工具调用与推理输出等能力，适用于对话、工具调用与多模态等场景。

    以下示例均为后续内容中会用到的模型部署命令：
    
    **Qwen2.5-7B**：

    ```bash
    vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen2.5-7b
    ```

    **Qwen3-4B**：

    ```bash
    vllm serve Qwen/Qwen3-4B \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3-4b
    ```

    **GLM-4.7-Flash**：

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

    **Qwen2.5-VL-7B**：

    ```bash
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen2.5-vl-7b
    ```

    **Qwen3-Embedding-4B**：

    ```bash
    vllm serve Qwen/Qwen3-Embedding-4B \
    --task embed \
    --served-model-name qwen3-embedding-4b \
    --host 0.0.0.0 --port 8000
    ```
    服务地址为 `http://localhost:8000/v1`。

