from langchain_core.messages import AIMessage, ToolCall


from langchain_dev_utils.tool_calling import has_tool_calling, parse_tool_calling


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


def test_parse_tool_call():
    message = AIMessage(
        content="Hello",
        tool_calls=[
            ToolCall(id="1", name="tool1", args={"arg1": "value1"}),
            ToolCall(id="2", name="tool2", args={"arg2": "value2"}),
        ],
    )
    assert parse_tool_calling(message) == [
        ("tool1", {"arg1": "value1"}),
        ("tool2", {"arg2": "value2"}),
    ]
    assert parse_tool_calling(message, first_tool_call_only=True) == (
        "tool1",
        {"arg1": "value1"},
    )
