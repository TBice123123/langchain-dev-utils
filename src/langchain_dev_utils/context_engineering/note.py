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

    This function creates a tool that allows agents to write notes and store them
    in the state. The notes are stored in a dictionary with the note name as the key
    and the content as the value.

    Args:
        name: The name of the tool. Defaults to "write_note".
        description: The description of the tool. Uses default description if not provided.
        message_key: The key of the message to be updated. Defaults to "messages".

    Returns:
        BaseTool: The tool for writing notes.

    Example:
        Basic usage:
        >>> from langchain_dev_utils import create_write_note_tool
        >>> write_note = create_write_note_tool()
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
        notes = state.get("note", {})
        if file_name in notes:
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

    This function creates a tool that allows agents to list all available notes
    stored in the state. This is useful for discovering what notes have been
    created before querying or updating them.

    Args:
        name: The name of the tool. Defaults to "ls".
        description: The description of the tool. Uses default description if not provided.

    Returns:
        BaseTool: The tool for listing all the saved note names.

    Example:
        Basic usage:
        >>> from langchain_dev_utils import create_ls_tool
        >>> ls = create_ls_tool()
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
        notes = state.get("note", {})
        return list(notes.keys())

    return ls


def create_query_note_tool(
    name: Optional[str] = None, description: Optional[str] = None
) -> BaseTool:
    """Create a tool for querying the content of a note.

    This function creates a tool that allows agents to retrieve the content of
    a specific note by its name. This is useful for accessing previously stored
    information during the conversation.

    Args:
        name: The name of the tool. Defaults to "query_note".
        description: The description of the tool. Uses default description if not provided.

    Returns:
        BaseTool: The tool for querying the content of a note.

    Example:
        Basic usage:
        >>> from langchain_dev_utils import create_query_note_tool
        >>> query_note = create_query_note_tool()
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
        notes = state.get("note", {})
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

    This function creates a tool that allows agents to update the content of
    existing notes. The tool can replace either the first occurrence of the
    original content or all occurrences, depending on the replace_all parameter.

    Args:
        name: The name of the tool. Defaults to "update_note".
        description: The description of the tool. Uses default description if not provided.
        message_key: The key of the message to be updated. Defaults to "messages".

    Returns:
        BaseTool: The tool for updating notes.

    Example:
        Basic usage:
        >>> from langchain_dev_utils import create_update_note_tool
        >>> update_note_tool = create_update_note_tool()
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
        notes = state.get("note", {})
        if file_name not in notes:
            raise ValueError(f"Error: Note {file_name} not found")

        if origin_content not in notes.get(file_name, ""):
            raise ValueError(
                f"Error: Origin content {origin_content} not found in note {file_name}"
            )

        if replace_all:
            new_content = notes.get(file_name, "").replace(origin_content, new_content)
        else:
            new_content = notes.get(file_name, "").replace(
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
