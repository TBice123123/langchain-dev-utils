# 格式化序列

## 概述

用于将由 Document、Message 或字符串组成的列表格式化为单个文本字符串。具体函数为`format_sequence`。


## 使用示例

使用示例如下：

### Message

```python
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_dev_utils.message_convert import format_sequence

formated1 = format_sequence(
    [
        AIMessage(content="Hello1"),
        AIMessage(content="Hello2"),
        AIMessage(content="Hello3"),
    ]
)
print(formated1)
```

输出结果：

```
-Hello1
-Hello2
-Hello3
```

### Document

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

输出结果：

```
>content1
>content2
>content3
```

### String

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

输出结果：

```
>1. str1
>2. str2
>3. str3
```
