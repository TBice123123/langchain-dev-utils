from typing import Dict, List, Tuple, Union

from langchain_core.messages import AIMessage


def has_tool_calling(message: AIMessage) -> bool:
    """Check if a message contains tool calls.

    Args:
        message: Any message type to check for tool calls

    Returns:
        bool: True if message is an AIMessage with tool calls, False otherwise
    """
    if (
        isinstance(message, AIMessage)
        and hasattr(message, "tool_calls")
        and len(message.tool_calls) > 0
    ):
        return True
    return False


def parse_tool_calling(
    message: AIMessage, first_tool_call_only: bool = False
) -> Union[Tuple[str, dict], List[Tuple[str, Dict]]]:
    """Parse a tool call from a message.

    Args:
        message: Any message type to parse for tool calls
        first_tool_call_only: If True, only the first tool call will be parsed

    Returns:
        Union[Tuple[str, dict], List[Tuple[str, Dict]]]: The tool call name and args
    """

    if first_tool_call_only:
        return (message.tool_calls[0]["name"], message.tool_calls[0]["args"])
    return [(tool_call["name"], tool_call["args"]) for tool_call in message.tool_calls]
