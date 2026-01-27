# Sequence Formatting

## Overview

Used to format a list consisting of Documents, Messages, or strings into a single text string. The specific function is `format_sequence`.

## Usage Examples

### Message

#### Code Example

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_dev_utils.message_convert import format_sequence

formated1 = format_sequence(
    [
        SystemMessage(content="You are a weather query assistant"),
        HumanMessage(content="Check the weather in London and San Francisco"),
        AIMessage(
            content="I will use the get_weather tool to check the weather for these two cities",
            tool_calls=[
                {"name": "get_weather", "args": {"location": "London"}, "id": "123"},
                {"name": "get_weather", "args": {"location": "San Francisco"}, "id": "456"},
            ],
        ),
        ToolMessage(
            content="The weather in London is 25 degrees Celsius, and the weather in San Francisco is 22 degrees Celsius",
            tool_call_id="123",
        ),
        ToolMessage(
            content="The weather in London is 25 degrees Celsius, and the weather in San Francisco is 22 degrees Celsius",
            tool_call_id="456",
        ),
        AIMessage(
            content="Based on the tool call results, the weather in London is 25 degrees Celsius, and the weather in San Francisco is 22 degrees Celsius",
        ),
    ]
)
print(formated1)
```

#### Output Result

```
-System: You are a weather query assistant
-Human: Check the weather in London and San Francisco
-AI: I will use the get_weather tool to check the weather for these two cities
</think><tool_call>get_weather</arg_value>
<tool_call>get_weather</arg_value>
-Tool: The weather in London is 25 degrees Celsius, and the weather in San Francisco is 22 degrees Celsius
-Tool: The weather in London is 25 degrees Celsius, and the weather in San Francisco is 22 degrees Celsius
-AI: Based on the tool call results, the weather in London is 25 degrees Celsius, and the weather in San Francisco is 22 degrees Celsius
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