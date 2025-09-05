from langchain_core.messages import AIMessage, ToolCall
from langchain_dev_utils.has_tool_calling import has_tool_calling


def test_has_tool_calling():
    message = AIMessage(
        content="Hello",
        tool_calls=[ToolCall(id="1", name="tool1", args={"arg1": "value1"})],
    )
    assert has_tool_calling(message)

    message = AIMessage(content="Hello")
    assert not has_tool_calling(message)

    message = AIMessage(content="Hello", tool_calls=[])
    assert not has_tool_calling(message)
