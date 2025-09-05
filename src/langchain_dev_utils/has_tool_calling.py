from langchain_core.messages import AIMessage


def has_tool_calling(message: AIMessage):
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
