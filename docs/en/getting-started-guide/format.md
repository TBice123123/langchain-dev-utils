# Sequence Formatting

## Overview

Used to format a list consisting of Documents, Messages, or strings into a single text string. The specific function is `format_sequence`.

## Usage Examples

### Message

#### Use Cases

- Convert conversation history (system/human/ai/tool) into readable text, making it easy to inject into the next prompt.
- Debug printing: render a message list into a readable format for logs.

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
            content="The weather in London is 25 degrees Celsius",
            tool_call_id="123",
        ),
        ToolMessage(
            content="The weather in San Francisco is 22 degrees Celsius",
            tool_call_id="456",
        ),
        AIMessage(
            content="Based on the tool call results, the weather in London is 25 degrees Celsius, and the weather in San Francisco is 22 degrees Celsius",
        ),
    ]
    # The separator parameter defaults to "-" and with_num defaults to False; using defaults here
)
print(formated1)
```

#### Output Result

```
-System: You are a weather query assistant
-Human: Check the weather in London and San Francisco
-AI: I will use the get_weather tool to check the weather for these two cities
<tool_call>get_weather</tool_call>
<tool_call>get_weather</tool_call>
-Tool: The weather in London is 25 degrees Celsius
-Tool: The weather in San Francisco is 22 degrees Celsius
-AI: Based on the tool call results, the weather in London is 25 degrees Celsius, and the weather in San Francisco is 22 degrees Celsius
```

### Document

#### Use Cases

- RAG: format the retrieved `Document` list into a `context` text block and paste it directly into your prompt.

#### Code Example

```python hl_lines="11 12"
from langchain_core.documents import Document

from langchain_dev_utils.message_convert import format_sequence

format2 = format_sequence(
    [
        Document(page_content="[Source: Product Manual] Refund policy: refunds are allowed within 7 days."),
        Document(page_content="[Source: FAQ] Refunds usually take 1-3 business days to arrive."),
        Document(page_content="[Source: Support Guidelines] In disputes, apologize first and ask for the order ID."),
    ],
    separator=">",  # Set the separator to ">" so each line starts with this symbol
    with_num=True,  # Enable numbering so documents are labeled 1, 2, 3...
)
print(format2)
```

#### Output Result

```
>1. [Source: Product Manual] Refund policy: refunds are allowed within 7 days.
>2. [Source: FAQ] Refunds usually take 1-3 business days to arrive.
>3. [Source: Support Guidelines] In disputes, apologize first and ask for the order ID.
```

### String

#### Use Cases

- Format a set of bullet points (requirements, checklist items, todos, etc.) into multi-line text for prompt composition.

#### Code Example

```python
from langchain_dev_utils.message_convert import format_sequence

format3 = format_sequence(
    [
        "Answer the user's question only; do not add extra content.",
        "If uncertain, state your assumptions clearly.",
        "Use a Markdown list in the output.",
    ],
    separator=">",
    with_num=True,
)
print(format3)
```

#### Output Result

```
>1. Answer the user's question only; do not add extra content.
>2. If uncertain, state your assumptions clearly.
>3. Use a Markdown list in the output.
```
