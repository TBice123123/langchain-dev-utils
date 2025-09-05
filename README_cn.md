# LangChain Dev Utils(中文版)

本工具包旨在为使用 LangChain 和 LangGraph 开发大语言模型应用的开发者提供封装好的实用工具，帮助开发者更高效地进行开发工作。

## 安装和使用

1. 使用 pip

```bash
pip install -U langchain-dev-utils
```

2. 使用 poetry

```bash
poetry add langchain-dev-utils
```

3. 使用 uv

```bash
uv add langchain-dev-utils
```

## 功能模块

### 1. 扩展模型提供商

官方的 `init_chat_model` 函数虽然非常好用，但支持的模型提供商有限。本工具包提供了扩展模型加载功能，可以注册和使用更多模型提供商。

#### 核心函数

- `register_model_provider`: 注册模型提供商
- `load_chat_model`: 加载聊天模型

#### `register_model_provider` 参数说明

- `provider_name`: 提供商名称，需要自定义名称
- `chat_model`: ChatModel 类或者字符串，如果是字符串，必须是官方 `init_chat_model` 支持的提供商（例如 `openai`、`anthropic`）此时会调用`init_chat_model`函数
- `base_url`: 可选的基础 URL，在 `chat_model` 为字符串时建议传入

#### 使用示例

```python
from langchain_dev_utils.chat_model import register_model_provider, load_chat_model
from langchain_qwq import ChatQwen
from dotenv import load_dotenv

load_dotenv()

# 注册自定义模型提供商
register_model_provider("dashscope", ChatQwen)
register_model_provider("openrouter", "openai", base_url="https://openrouter.ai/api/v1")

# 加载模型
model = load_chat_model(model="dashscope:qwen-flash")
print(model.invoke("你好啊"))

model = load_chat_model(model="openrouter:moonshotai/kimi-k2-0905")
print(model.invoke("你好啊"))
```

**注意事项**：由于函数的底层实现是一个全局字典，**必须在应用启动时完成所有模型提供商的注册**，运行时不应再进行修改，否则可能引发多线程并发同步问题。

我们建议你将将`register_model_provider`放在应用的`__init__.py`文件中。

例如你有如下的 LangGraph 目录结构

```text
langgraph-project/
├── src
│   ├── __init__.py
│   └── graphs
│       ├── __init__.py # 在这里调用 register_model_provider
│       ├── graph1
│       └── graph2
```

### 2. 合并推理内容

提供合并推理模型返回的`reasoning_content`到 AI 消息的`content`中的功能。

#### 核心函数

- `convert_reasoning_content_for_ai_message`: 转换单个 AI 消息的推理内容
- `convert_reasoning_content_for_chunk_iterator`: 转换流式响应中消息块迭代器的推理内容
- `aconvert_reasoning_content_for_chunk_iterator`: 异步转换流式响应中消息块迭代器的推理内容

#### 使用示例

```python
# 同步处理推理内容
from langchain_dev_utils.content import convert_reasoning_content_for_ai_message

response = model.invoke("请解决这个数学问题")
converted_response = convert_reasoning_content_for_ai_message(response, think_tag=("<think>", "</think>"))

# 流式处理推理内容
from langchain_dev_utils.content import convert_reasoning_content_for_chunk_iterator

for chunk in convert_reasoning_content_for_chunk_iterator(model.stream("请解决这个数学问题"), think_tag=("<think>", "</think>")):
    print(chunk.content, end="", flush=True)
```

### 3. 拓展嵌入模型提供商

提供扩展的嵌入模型加载功能，类似于扩展模型提供商的功能。

#### 核心函数

- `register_embeddings_provider`: 注册嵌入模型提供商
- `load_embeddings`: 加载嵌入模型

#### 使用示例

```python
from langchain_dev_utils.embbedings import register_embeddings_provider, load_embeddings

# 注册嵌入模型提供商
register_embeddings_provider("openai", "openai", base_url="https://api.openai.com/v1")

# 加载嵌入模型
embeddings = load_embeddings("openai:text-embedding-ada-002")
```

**注意事项**：由于该函数的底层实现是同样是一个全局字典，**必须在应用启动时完成所有嵌入模型提供商的注册**，后续调用的时候不应再进行修改，否则可能引发多线程并发同步问题。

同样的我们建议你将`register_embeddings_provider`放在应用的`__init__.py`文件中。具体可以参考上文的`注册模型提供商`部分。

### 4. 工具调用检测

提供检测消息是否包含工具调用的简单函数。

#### 核心函数

- `has_tool_calling`: 检测消息是否包含工具调用

#### 使用示例

```python
from langchain_dev_utils.has_tool_calling import has_tool_calling

if has_tool_calling(message):
    # 处理工具调用逻辑
    pass
```

### 5.合并 AI 消息块

提供合并 AI 消息块的工具函数，用于将多个 AI 消息块合并为一个 AI 消息。

#### 核心函数

- `merge_ai_message_chunk`: 合并 AI 消息块

#### 使用示例

```python
from langchain_dev_utils.content import merge_ai_message_chunk

chunks = [
    AIMessageChunk(content="Chunk 1"),
    AIMessageChunk(content="Chunk 2"),
]
merged_message = merge_ai_message_chunk(chunks)
```

## Test

本项目目前所有的工具函数均通过测试，你也可以克隆本项目进行测试

```bash
git clone https://github.com/TBice123123/langchain-dev-utils.git
```

```bash
cd langchain-dev-utils
```

```bash
uv sync --group test
```

```bash
uv run pytest .
```
