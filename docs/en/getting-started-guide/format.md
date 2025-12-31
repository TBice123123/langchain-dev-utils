# Formatting Sequence

## Overview

Used to format a list consisting of Documents, Messages, or strings into a single text string. The specific function is `format_sequence`.

## Usage Examples

### Message

#### Code Example

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

#### Output Result

```
-Hello1
-Hello2
-Hello3
```

### Document

#### Code Example

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

#### Output Result

```
>content1
>content2
>content3
```

### String

#### Code Example

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

#### Output Result

```
>1. str1
>2. str2
>3. str3
```