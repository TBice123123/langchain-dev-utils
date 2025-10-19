from typing import Sequence, Union

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def format_sequence(
    inputs: Union[Sequence[Document], Sequence[BaseMessage], Sequence[str]],
    separator: str = "-",
    with_num: bool = False,
) -> str:
    """Convert a list of messages, documents, or strings into a formatted string.

    This function extracts text content from various types (e.g., HumanMessage, Document)
    and joins them into a single string. Optionally adds serial numbers and a custom
    separator between items.

    Args:
        inputs: A list of inputs. Supported types:
            - langchain_core.messages: HumanMessage, AIMessage, SystemMessage, ToolMessage
            - langchain_core.documents.Document
            - str
        separator: The separator used to join the items. Defaults to "-".
        with_num: If True, prefixes each item with a serial number (e.g., "1. Hello").
                  Defaults to False.

    Returns:
        A formatted string composed of the input contents, joined by `separator`.

    Example:
        Format messages with default separator:
        >>> from langchain_dev_utils.message_convert import format_sequence
        >>> from langchain_core.messages import HumanMessage, AIMessage
        >>> messages = [
        ...     HumanMessage(content="Hello, how are you?"),
        ...     AIMessage(content="I'm doing well, thank you!")
        ... ]
        >>> formatted = format_sequence(messages)
        >>> formatted

        Format with custom separator and numbering:
        >>> formatted = format_sequence(messages, separator="---", with_num=True)
        >>> formatted
    """
    if not inputs:
        return ""

    outputs = []

    for input_item in inputs:
        if isinstance(
            input_item, (HumanMessage, AIMessage, SystemMessage, ToolMessage)
        ):
            outputs.append(input_item.content)
        elif isinstance(input_item, Document):
            outputs.append(input_item.page_content)
        elif isinstance(input_item, str):
            outputs.append(input_item)
    if with_num:
        outputs = [f"{i + 1}. {output}" for i, output in enumerate(outputs)]

    str_ = "\n" + separator
    return separator + str_.join(outputs)
