
# 嵌入模型的创建与使用

## 创建嵌入模型类

与对话模型类类似，可以使用 `create_openai_compatible_embedding` 创建嵌入模型集成类。该函数接受以下参数：

| 参数 | 说明 |
|------|------|
| `embedding_provider` | 嵌入模型提供商名称，例如 `vllm`。必须以字母或数字开头，只能包含字母、数字和下划线，长度不超过 20 个字符。<br><br>**类型**: `str`<br>**必填**: 是 |
| `base_url` | 模型提供商默认 API 地址。<br><br>**类型**: `str`<br>**必填**: 否 |
| `embedding_model_cls_name` | 嵌入模型类名（需符合 Python 类名规范）。默认值为 `{Provider}Embeddings`（其中 `{Provider}` 为首字母大写的提供商名称）。<br><br>**类型**: `str`<br>**必填**: 否 |

同样，我们使用 `create_openai_compatible_embedding` 来集成 vLLM 的嵌入模型。

```python hl_lines="4 5 6"
from langchain_dev_utils.embeddings.adapters import create_openai_compatible_embedding

VLLMEmbeddings = create_openai_compatible_embedding(
    embedding_provider="vllm",
    base_url="http://localhost:8000/v1",
    embedding_model_cls_name="VLLMEmbeddings",
)

embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_query("你好"))
```

`base_url` 也可以省略。未传入时，本库会默认读取环境变量 `VLLM_API_BASE`：

```bash
export VLLM_API_BASE="http://localhost:8000/v1"
```

此时代码可以省略 `base_url`：

```python hl_lines="4 5"
from langchain_dev_utils.embeddings.adapters import create_openai_compatible_embedding

VLLMEmbeddings = create_openai_compatible_embedding(
    embedding_provider="vllm",
    embedding_model_cls_name="VLLMEmbeddings",
)

embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_query("你好"))
```

**注意**：上述代码成功运行的前提是已配置环境变量 `VLLM_API_KEY`。虽然 vLLM 本身不要求 API Key，但嵌入模型类初始化时需要传入，因此请先设置该变量，例如：

```bash
export VLLM_API_KEY=vllm_api_key
```

## 使用嵌入模型类

这里使用前面创建好的 `VLLMEmbeddings` 类来初始化嵌入模型实例。

### 向量化查询

```python
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_query("你好"))
```

同样，也支持异步调用：

```python
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
res = await embedding.aembed_query("你好")
print(res)
```

### 向量化字符串列表

```python
documents = ["你好", "你好，我是张三"]
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
print(embedding.embed_documents(documents))
```
同样，也支持异步调用：

```python
documents = ["你好", "你好，我是张三"]
embedding = VLLMEmbeddings(model="qwen3-embedding-4b")
res = await embedding.aembed_documents(documents)
print(res)
```

!!! warning "嵌入模型兼容性说明"
    兼容 OpenAI 的嵌入 API 通常表现出较好的兼容性，但仍需注意以下差异点：

    1. `check_embedding_ctx_length`：仅在使用官方 OpenAI 嵌入服务时设为 `True`；其余嵌入模型一律设为 `False`。

    2. `dimensions`：若模型支持自定义向量维度（如 1024、4096），可直接传入该参数。

    3. `chunk_size`：为单次API调用中能处理的文本数量上限。例如`chunk_size`大小为10，意味着一次请求最多可传入10个文本进行向量化。
    
    4. 单文本 token 上限：无法通过参数控制，需在预处理分块阶段自行保证。


!!! warning "注意"
    同样，该函数底层使用 [pydantic 的 create_model](https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation) 创建嵌入模型类，会带来一定的性能开销。建议在项目启动阶段创建好集成类，后续避免动态创建。


!!! success "最佳实践"
    接入 OpenAI 兼容 API 的嵌入模型提供商时，可以直接使用 `langchain-openai` 的 `OpenAIEmbeddings`，并通过 `base_url` 与 `api_key` 指向你的提供商服务，嵌入模型API的兼容性通常更好：多数情况下直接使用 `OpenAIEmbeddings` 并将 `check_embedding_ctx_length=False` 即可。