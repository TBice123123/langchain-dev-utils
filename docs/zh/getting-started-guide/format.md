# 格式化序列

## 概述

用于将由 Document、Message 或字符串组成的列表格式化为单个文本字符串。具体函数为 `format_sequence`。

## 使用示例

### Message

#### 应用场景

- 将对话历史（system/human/ai/tool）压成一段可读文本，便于注入到下一轮提示词。
- 打印调试：将消息列表打印成可读格式，方便在日志里查看。

#### 代码示例

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_dev_utils.message_convert import format_sequence

formated1 = format_sequence(
    [
        SystemMessage(content="你是一个天气查询助手"),
        HumanMessage(content="查询一下伦敦和旧金山的天气"),
        AIMessage(
            content="我将使用get_weather工具查询这两个城市的天气",
            tool_calls=[
                {"name": "get_weather", "args": {"location": "伦敦"}, "id": "123"},
                {"name": "get_weather", "args": {"location": "旧金山"}, "id": "456"},
            ],
        ),
        ToolMessage(
            content="伦敦的天气是25摄氏度",
            tool_call_id="123",
        ),
        ToolMessage(
            content="旧金山的天气是22摄氏度",
            tool_call_id="456",
        ),
        AIMessage(
            content="根据工具调用的结果，伦敦的天气是25摄氏度，旧金山的天气是22摄氏度",
        ),
    ]
)
print(formated1)
```

#### 输出结果

```
-System: 你是一个天气查询助手
-Human: 查询一下伦敦和旧金山的天气
-AI: 我将使用get_weather工具查询这两个城市的天气
<tool_call>get_weather</tool_call>
<tool_call>get_weather</tool_call>
-Tool: 伦敦的天气是25摄氏度
-Tool: 旧金山的天气是22摄氏度
-AI: 根据工具调用的结果，伦敦的天气是25摄氏度，旧金山的天气是22摄氏度
```

### Document

#### 应用场景

- RAG：把检索返回的 `Document` 列表格式化为一段 `context` 文本，直接拼进提示词。

#### 代码示例

```python
from langchain_core.documents import Document

from langchain_dev_utils.message_convert import format_sequence

format2 = format_sequence(
    [
        Document(page_content="【来源: 产品手册】退款政策：7 天内可无理由退款。"),
        Document(page_content="【来源: FAQ】退款到账一般需要 1-3 个工作日。"),
        Document(page_content="【来源: 客服规范】遇到争议先致歉并引导提交订单号。"),
    ],
    separator=">",
    with_num=True,
)
print(format2)
```

#### 输出结果

```
>1. 【来源: 产品手册】退款政策：7 天内可无理由退款。
>2. 【来源: FAQ】退款到账一般需要 1-3 个工作日。
>3. 【来源: 客服规范】遇到争议先致歉并引导提交订单号。
```

### String

#### 应用场景

- 将一组要点（需求点、检查项、待办列表等）格式化成多行文本，方便拼进提示词。

#### 代码示例

```python
from langchain_dev_utils.message_convert import format_sequence

format3 = format_sequence(
    [
        "只回答用户问题，不要扩展",
        "不确定时明确说明假设",
        "输出使用 Markdown 列表",
    ],
    separator=">",
    with_num=True,
)
print(format3)
```

#### 输出结果

```
>1. 只回答用户问题，不要扩展
>2. 不确定时明确说明假设
>3. 输出使用 Markdown 列表
```
