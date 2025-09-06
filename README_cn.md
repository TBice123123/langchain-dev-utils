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

目前分为以下三个主模块

### 1.实例化模型对象

官方的`init_chat_model`和`init_embeddings`函数虽然使用便捷，但支持的模型提供商相对有限。为此，我们提供了`register_model_provider`和`register_embeddings_provider`函数，通过统一的注册和加载机制，让开发者能够灵活注册任意模型提供商，实现更广泛的模型支持，同时利用`load_chat_model`和`load_embeddings`保持与官方函数同样简洁的使用体验。

#### （1）ChatModel 类

**核心函数**

- `register_model_provider`: 注册模型提供商
- `load_chat_model`: 加载聊天模型

**`register_model_provider` 参数说明**

- `provider_name`: 提供商名称，需要自定义名称
- `chat_model`: ChatModel 类或者字符串，如果是字符串，必须是官方 `init_chat_model` 支持的提供商（例如 `openai`、`anthropic`）此时会调用`init_chat_model`函数
- `base_url`: 可选的基础 URL，在 `chat_model` 为字符串时建议传入

**`load_chat_model` 参数说明**

- `model`: 模型名称，格式为`model_name`或者`provider_name:model_name`
- `model_provider`: 可选的模型提供商名称，如果不传，则需要在 model 中包含提供商名称
- `kwargs`: 可选的剩余模型参数，例如 temperature、api_key、stop 等。
  上面这三个参数和官方的`init_chat_model`函数的参数一致。

- **注意**：目前暂且不支持传入`configurable_fields`和`config_prefix`参数。

**使用示例**

```python
from langchain_dev_utils import register_model_provider, load_chat_model
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

**建议**：我们建议你将将`register_model_provider`放在应用的`__init__.py`文件中。

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

#### (2) Embeddings 类

**核心函数**

- `register_embeddings_provider`: 注册嵌入模型提供商
- `load_embeddings`: 加载嵌入模型

**`register_embeddings_provider` 参数说明**

- `provider_name`: 提供商名称，需要自定义名称
- `embeddings_model`: Embeddings 类或者字符串，如果是字符串，必须是官方 `init_embeddings` 支持的提供商（例如 `openai`、`anthropic`）此时会调用`init_embeddings`函数
- `base_url`: 可选的基础 URL，在 `embeddings` 为字符串时建议传入

**`load_embeddings` 参数说明**

- `model`: 模型名称，格式为`model_name`或者`provider_name:model_name`
- `provider`: 可选的模型提供商名称，如果不传，则需要在 model 中包含提供商名称
- `kwargs`: 可选的剩余模型参数，例如 chunk_size、api_key、dimensions 等。
  上面这三个参数和官方的`init_embeddings`函数的参数一致。

**使用示例**

```python
from langchain_dev_utils import register_embeddings_provider, load_embeddings

register_embeddings_provider(
    "dashscope", "openai", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

embeddings = load_embeddings("dashscope:text-embedding-v4")

print(embeddings.embed_query("hello world"))
```

**注意事项**：由于该函数的底层实现是同样是一个全局字典，**必须在应用启动时完成所有嵌入模型提供商的注册**，后续调用的时候不应再进行修改，否则可能引发多线程并发同步问题。

同样的我们建议你将`register_embeddings_provider`放在应用的`__init__.py`文件中。具体可以参考上文的`注册模型提供商`部分。

### Message 类处理

#### (1) 合并推理内容

提供合并推理模型返回的`reasoning_content`到 AI 消息的`content`中的功能。

**核心函数**

- `convert_reasoning_content_for_ai_message`: 将 AIMessage 的 reasoning_content 合并到 content 中
- `convert_reasoning_content_for_chunk_iterator`: 将流式响应中消息块迭代器的 reasoning_content 合并到 content 中
- `aconvert_reasoning_content_for_chunk_iterator`: 异步流式的`convert_reasoning_content_for_chunk_iterator`

**参数说明**

- `model_response`：模型的 AI 消息响应
- `think_tag`：包含推理内容的开始和结束标签的元组

**使用示例**

```python
# 同步处理推理内容
from typing import cast
from langchain_dev_utils import convert_reasoning_content_for_ai_message
from langchain_core.messages import AIMessage

# 流式处理推理内容
from langchain_dev_utils import convert_reasoning_content_for_chunk_iterator

response = model.invoke("你好")
converted_response = convert_reasoning_content_for_ai_message(
    cast(AIMessage, response), think_tag=("<think>", "</think>")
)
print(converted_response.content)

for chunk in convert_reasoning_content_for_chunk_iterator(
    model.stream("你好"), think_tag=("<think>", "</think>")
):
    print(chunk.content, end="", flush=True)
```

#### (2) 合并 AI 消息块

提供合并 AI 消息块的工具函数，用于将多个 AI 消息块合并为一个 AI 消息。

**核心函数**

- `merge_ai_message_chunk`: 合并 AI 消息块

**参数说明**

- `chunks`：AI 消息块列表

**使用示例**

```python
from langchain_dev_utils import merge_ai_message_chunk

chunks = []
for chunk in model.stream("你好"):
    chunks.append(chunk)

merged_message = merge_ai_message_chunk(chunks)
print(merged_message)

```

#### (3) 检测消息是否包含工具调用

提供检测消息是否包含工具调用的简单函数。

**核心函数**

- `has_tool_calling`: 检测消息是否包含工具调用

**参数说明**

- `message`：AIMessage 消息

**使用示例**

```python
import datetime

from langchain_core.tools import tool

from langchain_dev_utils import has_tool_calling
from langchain_core.messages import AIMessage
from typing import cast


@tool
def get_current_time() -> str:
    """获取当前时间戳"""
    return str(datetime.datetime.now().timestamp())


response = model.bind_tools([get_current_time]).invoke("查询当前时间")

print(has_tool_calling(cast(AIMessage, response)))

```

#### （4）解析工具调用参数

提供解析工具调用参数的工具函数，用于从消息中提取工具调用参数。

**核心函数**

- `parse_tool_calling`: 解析工具调用参数

**参数说明**

- `message`：AIMessage 消息
- `first_tool_call_only`：是否只解析第一个工具调用参数，如果为 true 返回值将是一个二元组，为 false 则是返回一个多个二元组的列表

**使用示例**

```python
import datetime

from langchain_core.tools import tool

from langchain_dev_utils import has_tool_calling, parse_tool_calling
from langchain_core.messages import AIMessage
from typing import cast


@tool
def get_current_time() -> str:
    """获取当前时间戳"""
    return str(datetime.datetime.now().timestamp())


response = model.bind_tools([get_current_time]).invoke("查询当前时间")

if has_tool_calling(cast(AIMessage, response)):
    name, args = parse_tool_calling(
        cast(AIMessage, response), first_tool_call_only=True
    )
    print(name, args)
```

#### （5）格式化消息

将 Document、Message、字符串组成的列表格式化为字符串。

**核心函数**

- `message_format`: 格式化消息

**参数说明**

- inputs: 一个包含以下类型的列表，支持的类型有： - langchain_core.messages: HumanMessage, AIMessage, SystemMessage, ToolMessage - langchain_core.documents.Document - str
- separator: 用于连接内容的分隔符。默认为 "-"。
- with_num: 如果为 True，则在每个内容前添加一个序列号（例如，"1. Hello"）。默认为 False。

**使用示例**

```python
from langchain_dev_utils import message_format
from langchain_core.documents import Document

messages = [
    Document(page_content="Document 1"),
    Document(page_content="Document 2"),
    Document(page_content="Document 3"),
    Document(page_content="Document 4"),
]
formatted_messages = message_format(messages, separator="\n", with_num=True)
print(formatted_messages)

```

### 3.Tool 增强

#### （1）为工具调用增加 interrupt

提供为工具调用增加 human-in-the-loop review 支持的工具函数，用于在工具调用时增加 human-in-the-loop review 功能。

**核心函数**

- `human_in_the_loop`: 为工具调用增加 human-in-the-loop review
- `human_in_the_loop_async`: 为工具调用增加 human-in-the-loop review(异步)

**参数说明**

- func: 要装饰的函数。**不要直接传递此参数。**
- interrupt_config: 人工中断的配置。

**使用示例**

```python
from langchain_dev_utils import human_in_the_loop
from langchain_core.tools import tool
import datetime

@human_in_the_loop
@tool #也可以不写tool
def get_current_time() -> str:
    """获取当前时间戳"""
    return str(datetime.datetime.now().timestamp())
```

## 测试

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
