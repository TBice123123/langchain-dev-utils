# 格式化序列

## 概述

用于将由 Document、Message 或字符串组成的列表格式化为单个文本字符串。具体函数为 `format_sequence`。

## 使用示例

### Message

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
            content="伦敦的天气是25摄氏度，旧金山的天气是22摄氏度",
            tool_call_id="123",
        ),
        ToolMessage(
            content="伦敦的天气是25摄氏度，旧金山的天气是22摄氏度",
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
-Tool: 伦敦的天气是25摄氏度，旧金山的天气是22摄氏度
-Tool: 伦敦的天气是25摄氏度，旧金山的天气是22摄氏度
-AI: 根据工具调用的结果，伦敦的天气是25摄氏度，旧金山的天气是22摄氏度
```

### Document

#### 代码示例

```python
format2 = format_sequence(
    [
        Document(page_content="content1"),
        Document(page_content="content2"),
        Document(page_content="content3"),
    ],
    separator=">",
)
print(format2)
```

#### 输出结果

```
>content1
>content2
>content3
```

### String

#### 代码示例

```python
format3 = format_sequence(
    [
        "str1",
        "str2",
        "str3",
    ],
    separator=">",
    with_num=True,
)
print(format3)
```

#### 输出结果

```
>1. str1
>2. str2
>3. str3
```
