from typing import Annotated, Optional
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command
from typing_extensions import TypedDict

_DEFAULT_WRITE_NOTE_DESCRIPTION = """A tool for writing notes.
Parameters:
content: str, the content of the note
"""

_DEFAULT_LS_DESCRIPTION = """List all the saved note names."""


def note_reducer(left: dict | None, right: dict | None):
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right}


class NoteStateMixin(TypedDict):
    note: Annotated[dict[str, str], note_reducer]


def create_write_note_tool(
    name: Optional[str] = None, description: Optional[str] = None
) -> BaseTool:
    try:
        from langchain.agents.tool_node import InjectedState  # type: ignore
    except ImportError:
        from langgraph.prebuilt.tool_node import InjectedState

    @tool(
        name_or_callable=name or "write_note",
        description=description or _DEFAULT_WRITE_NOTE_DESCRIPTION,
    )
    def write_note(
        file_name: Annotated[str, "the name of the note"],
        content: Annotated[str, "the content of the note"],
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[NoteStateMixin, InjectedState],
    ):
        if file_name in state["note"] if "note" in state else {}:
            notes = state["note"] if "note" in state else {}
            file_name = file_name + "_" + str(len(notes[file_name]))

        return Command(
            update={
                "note": {file_name: content},
                "messages": [
                    ToolMessage(
                        content=f"note {file_name} written successfully, content is {content}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return write_note


def create_ls_tool(
    name: Optional[str] = None, description: Optional[str] = None
) -> BaseTool:
    try:
        from langchain.agents.tool_node import InjectedState  # type: ignore
    except ImportError:
        from langgraph.prebuilt.tool_node import InjectedState

    @tool(
        name_or_callable=name or "ls",
        description=description or "List all the saved note names.",
    )
    def ls(state: Annotated[NoteStateMixin, InjectedState]):
        notes = state["note"] if "note" in state else {}
        return list(notes.keys())

    return ls


def create_query_note_tool(
    name: Optional[str] = None, description: Optional[str] = None
) -> BaseTool:
    try:
        from langchain.agents.tool_node import InjectedState  # type: ignore
    except ImportError:
        from langgraph.prebuilt.tool_node import InjectedState

    @tool(
        name_or_callable=name or "query_note",
        description=description or "Query the content of a note.",
    )
    def query_note(file_name: str, state: Annotated[NoteStateMixin, InjectedState]):
        notes = state["note"] if "note" in state else {}
        return notes.get(file_name, "not found")

    return query_note
