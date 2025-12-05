# Formatting List Content

> [!NOTE]
>
> **Feature Overview**: Provides functionality to format common list content into a single string.
>
> **Prerequisites**: Understanding of langchain's [Message](https://docs.langchain.com/oss/python/langchain/messages).
>
> **Estimated Reading Time**: 2 minutes

## Overview

Used to format a list composed of Document, Message, or string into a single text string. The specific function is `format_sequence`.
Its parameters are as follows:

<Params
name="inputs"
type="list[Message | Document | string]"
description="A list containing any of the following types: langchain_core.messages, langchain_core.documents.Document, str"
:required="true"
:default="null"
/>
<Params
name="separator"
type="string"
description="String used to connect content"
:required="false"
default="'-'"
/>
<Params
name="with_num"
type="bool"
description="If True, add a numeric prefix to each item (e.g., '1. Hello')"
:required="false"
:default="false"
/>

## Usage Examples

Usage examples are as follows:

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

Output result:

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

Output result:

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

Output result:

```
>1. str1
>2. str2
>3. str3
```