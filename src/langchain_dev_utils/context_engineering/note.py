from typing import Annotated, Optional
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command
from typing_extensions import TypedDict

_DEFAULT_WRITE_NOTE_DESCRIPTION = """
A tool for writing notes.

Args:
    content: The content of the note
"""

_DEFAULT_LS_DESCRIPTION = """List all the saved note names."""


_DEFAULT_QUERY_NOTE_DESCRIPTION = """
Query the content of a note.

Args:
    file_name: The name of the note
"""

_DEFAULT_UPDATE_NOTE_DESCRIPTION = """
Update the content of a note.

Args:
    file_name: The name of the note
    origin_content: The original content of the note, must be a content in the note
    new_content: The new content of the note
    replace_all: Whether to replace all the origin content
"""


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
    name: Optional[str] = None,
    description: Optional[str] = None,
    message_key: Optional[str] = None,
) -> BaseTool:
    """Create a tool for writing notes.

    Args:
        name: The name of the tool.
        description: The description of the tool.
        message_key: The key of the message to be updated.
    Returns:
        BaseTool: The tool for writing notes.
    """
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

        msg_key = message_key or "messages"
        return Command(
            update={
                "note": {file_name: content},
                msg_key: [
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
    """Create a tool for listing all the saved note names.

    Args:
        name: The name of the tool.
        description: The description of the tool.
    Returns:
        BaseTool: The tool for listing all the saved note names.
    """
    try:
        from langchain.agents.tool_node import InjectedState  # type: ignore
    except ImportError:
        from langgraph.prebuilt.tool_node import InjectedState

    @tool(
        name_or_callable=name or "ls",
        description=description or _DEFAULT_LS_DESCRIPTION,
    )
    def ls(state: Annotated[NoteStateMixin, InjectedState]):
        notes = state["note"] if "note" in state else {}
        return list(notes.keys())

    return ls


def create_query_note_tool(
    name: Optional[str] = None, description: Optional[str] = None
) -> BaseTool:
    """Create a tool for querying the content of a note.

    Args:
        name: The name of the tool.
        description: The description of the tool.
    Returns:
        BaseTool: The tool for querying the content of a note.
    """
    try:
        from langchain.agents.tool_node import InjectedState  # type: ignore
    except ImportError:
        from langgraph.prebuilt.tool_node import InjectedState

    @tool(
        name_or_callable=name or "query_note",
        description=description or _DEFAULT_QUERY_NOTE_DESCRIPTION,
    )
    def query_note(file_name: str, state: Annotated[NoteStateMixin, InjectedState]):
        notes = state["note"] if "note" in state else {}
        if file_name not in notes:
            raise ValueError(f"Error: Note {file_name} not found")

        content = notes.get(file_name)

        if not content or content.strip() == "":
            raise ValueError(f"Error: Note {file_name} is empty")

        return content

    return query_note


def create_update_note_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    message_key: Optional[str] = None,
) -> BaseTool:
    """Create a tool for updating notes.

    Args:
        name: The name of the tool.
        description: The description of the tool.
        message_key: The key of the message to be updated.
    Returns:
        BaseTool: The tool for writing notes.
    """
    try:
        from langchain.agents.tool_node import InjectedState  # type: ignore
    except ImportError:
        from langgraph.prebuilt.tool_node import InjectedState

    @tool(
        name_or_callable=name or "update_note",
        description=description or _DEFAULT_UPDATE_NOTE_DESCRIPTION,
    )
    def update_note(
        file_name: Annotated[str, "the name of the note"],
        origin_content: Annotated[str, "the original content of the note"],
        new_content: Annotated[str, "the new content of the note"],
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[NoteStateMixin, InjectedState],
        replace_all: Annotated[bool, "replace all the origin content"] = False,
    ):
        msg_key = message_key or "messages"
        note = state["note"] if "note" in state else {}
        if file_name not in note:
            raise ValueError(f"Error: Note {file_name} not found")

        if origin_content not in note.get(file_name, ""):
            raise ValueError(
                f"Error: Origin content {origin_content} not found in note {file_name}"
            )

        if replace_all:
            new_content = note.get(file_name, "").replace(origin_content, new_content)
        else:
            new_content = note.get(file_name, "").replace(
                origin_content, new_content, 1
            )
        return Command(
            update={
                "note": {file_name: new_content},
                msg_key: [
                    ToolMessage(
                        content=f"note {file_name} updated successfully, content is {new_content}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return update_note
