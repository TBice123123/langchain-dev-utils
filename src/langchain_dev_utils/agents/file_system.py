from typing import Annotated, Literal, Optional
import warnings

from langchain.tools import BaseTool, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing_extensions import TypedDict

warnings.warn(
    "langchain_dev_utils.agents.file_system is deprecated, and it will be removed in a future version. Please use middleware in deepagents instead.",
    DeprecationWarning,
)

_DEFAULT_WRITE_FILE_DESCRIPTION = """
A tool for writing files.

Args:
    content: The content of the file
"""

_DEFAULT_LS_DESCRIPTION = """List all the saved file names."""


_DEFAULT_QUERY_FILE_DESCRIPTION = """
Query the content of a file.

Args:
    file_name: The name of the file
"""

_DEFAULT_UPDATE_FILE_DESCRIPTION = """
Update the content of a file.

Args:
    file_name: The name of the file
    origin_content: The original content of the file, must be a content in the file
    new_content: The new content of the file
    replace_all: Whether to replace all the origin content
"""


def file_reducer(left: dict | None, right: dict | None):
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right}


class FileStateMixin(TypedDict):
    file: Annotated[dict[str, str], file_reducer]


def create_write_file_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    message_key: Optional[str] = None,
) -> BaseTool:
    """Create a tool for writing files.

    This function creates a tool that allows agents to write files and store them
    in the state. The files are stored in a dictionary with the file name as the key
    and the content as the value.

    Args:
        name: The name of the tool. Defaults to "write_file".
        description: The description of the tool. Uses default description if not provided.
        message_key: The key of the message to be updated. Defaults to "messages".

    Returns:
        BaseTool: The tool for writing files.

    Example:
        Basic usage:
        >>> from langchain_dev_utils.agents.file_system import create_write_file_tool
        >>> write_file = create_write_file_tool()
    """

    @tool(
        name_or_callable=name or "write_file",
        description=description or _DEFAULT_WRITE_FILE_DESCRIPTION,
    )
    def write_file(
        file_name: Annotated[str, "the name of the file"],
        content: Annotated[str, "the content of the file"],
        runtime: ToolRuntime,
        write_mode: Annotated[
            Literal["write", "append"], "the write mode of the file"
        ] = "write",
    ):
        files = runtime.state.get("file", {})
        if write_mode == "append":
            content = files.get(file_name, "") + content
        if write_mode == "write" and file_name in files:
            # if the file already exists, append a suffix to the file name when write_mode is "write"
            file_name = file_name + "_" + str(len(files[file_name]))
        msg_key = message_key or "messages"
        return Command(
            update={
                "file": {file_name: content},
                msg_key: [
                    ToolMessage(
                        content=f"file {file_name} written successfully, content is {content}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return write_file


def create_ls_file_tool(
    name: Optional[str] = None, description: Optional[str] = None
) -> BaseTool:
    """Create a tool for listing all the saved file names.

    This function creates a tool that allows agents to list all available files
    stored in the state. This is useful for discovering what files have been
    created before querying or updating them.

    Args:
        name: The name of the tool. Defaults to "ls".
        description: The description of the tool. Uses default description if not provided.

    Returns:
        BaseTool: The tool for listing all the saved file names.

    Example:
        Basic usage:
        >>> from langchain_dev_utils.agents.file_system import create_ls_file_tool
        >>> ls = create_ls_file_tool()
    """

    @tool(
        name_or_callable=name or "ls",
        description=description or _DEFAULT_LS_DESCRIPTION,
    )
    def ls(runtime: ToolRuntime):
        files = runtime.state.get("file", {})
        return list(files.keys())

    return ls


def create_query_file_tool(
    name: Optional[str] = None, description: Optional[str] = None
) -> BaseTool:
    """Create a tool for querying the content of a file.

    This function creates a tool that allows agents to retrieve the content of
    a specific file by its name. This is useful for accessing previously stored
    information during the conversation.

    Args:
        name: The name of the tool. Defaults to "query_file".
        description: The description of the tool. Uses default description if not provided.

    Returns:
        BaseTool: The tool for querying the content of a file.

    Example:
        Basic usage:
        >>> from langchain_dev_utils.agents.file_system import create_query_file_tool
        >>> query_file = create_query_file_tool()
    """

    @tool(
        name_or_callable=name or "query_file",
        description=description or _DEFAULT_QUERY_FILE_DESCRIPTION,
    )
    def query_file(file_name: str, runtime: ToolRuntime):
        files = runtime.state.get("file", {})
        if file_name not in files:
            raise ValueError(f"Error: File {file_name} not found")

        content = files.get(file_name)

        if not content or content.strip() == "":
            raise ValueError(f"Error: File {file_name} is empty")

        return content

    return query_file


def create_update_file_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    message_key: Optional[str] = None,
) -> BaseTool:
    """Create a tool for updating files.

    This function creates a tool that allows agents to update the content of
    existing files. The tool can replace either the first occurrence of the
    original content or all occurrences, depending on the replace_all parameter.

    Args:
        name: The name of the tool. Defaults to "update_file".
        description: The description of the tool. Uses default description if not provided.
        message_key: The key of the message to be updated. Defaults to "messages".

    Returns:
        BaseTool: The tool for updating files.

    Example:
        Basic usage:
        >>> from langchain_dev_utils.agents.file_system import create_update_file_tool
        >>> update_file_tool = create_update_file_tool()
    """

    @tool(
        name_or_callable=name or "update_file",
        description=description or _DEFAULT_UPDATE_FILE_DESCRIPTION,
    )
    def update_file(
        file_name: Annotated[str, "the name of the file"],
        origin_content: Annotated[str, "the original content of the file"],
        new_content: Annotated[str, "the new content of the file"],
        runtime: ToolRuntime,
        replace_all: Annotated[bool, "replace all the origin content"] = False,
    ):
        msg_key = message_key or "messages"
        files = runtime.state.get("file", {})
        if file_name not in files:
            raise ValueError(f"Error: File {file_name} not found")

        if origin_content not in files.get(file_name, ""):
            raise ValueError(
                f"Error: Origin content {origin_content} not found in file {file_name}"
            )

        if replace_all:
            new_content = files.get(file_name, "").replace(origin_content, new_content)
        else:
            new_content = files.get(file_name, "").replace(
                origin_content, new_content, 1
            )
        return Command(
            update={
                "file": {file_name: new_content},
                msg_key: [
                    ToolMessage(
                        content=f"file {file_name} updated successfully, content is {new_content}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return update_file
