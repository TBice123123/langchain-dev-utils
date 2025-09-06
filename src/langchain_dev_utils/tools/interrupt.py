from typing import Any, Callable, Optional, Union, overload

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool as create_tool
from langgraph.prebuilt.interrupt import HumanInterrupt, HumanInterruptConfig
from langgraph.types import interrupt


@overload
def human_in_the_loop(
    func: Callable,
) -> BaseTool:
    """
    Usage: @human_in_the_loop_sync
    """
    ...


@overload
def human_in_the_loop(
    *,
    interrupt_config: Optional[HumanInterruptConfig] = None,
) -> Callable[[Callable], BaseTool]:
    """
    Usage: @human_in_the_loop_sync(interrupt_config={...})
    """
    ...


@overload
def human_in_the_loop_async(
    func: Callable,
) -> BaseTool:
    """
    Usage: @human_in_the_loop_async
    """
    ...


@overload
def human_in_the_loop_async(
    *,
    interrupt_config: Optional[HumanInterruptConfig] = None,
) -> Callable[[Callable], BaseTool]:
    """
    Usage: @human_in_the_loop_async(interrupt_config={...})
    """
    ...


def human_in_the_loop(
    func: Optional[Callable] = None,
    *,
    interrupt_config: Optional[HumanInterruptConfig] = None,
) -> Union[Callable[[Callable], BaseTool], BaseTool]:
    """
    A decorator that adds human-in-the-loop review support to a synchronous tool.

    Supports both syntaxes:
        @human_in_the_loop
        @human_in_the_loop(interrupt_config={...})

    Args:
        func: The function to decorate. **Do not pass this directly.**
        interrupt_config: Configuration for the human interrupt.

    Returns:
        If `func` is provided, returns the decorated BaseTool.
        If `func` is None, returns a decorator that will decorate the target function.
    """

    def _default_config() -> HumanInterruptConfig:
        return {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
            "allow_ignore": True,
        }

    def decorator(target_func: Callable) -> BaseTool:
        """The actual decorator that wraps the target function."""
        if not isinstance(target_func, BaseTool):
            tool_obj = create_tool(target_func)
        else:
            tool_obj = target_func

        final_config = interrupt_config or _default_config()

        @create_tool(
            tool_obj.name,
            description=tool_obj.description,
            args_schema=tool_obj.args_schema,
        )
        def tool_with_human_review(config: RunnableConfig, **tool_input: Any) -> Any:
            request: HumanInterrupt = {
                "action_request": {
                    "action": tool_obj.name,
                    "args": tool_input,
                },
                "config": final_config,
                "description": f"Please review tool call: {tool_obj.name}",
            }

            response = interrupt([request])
            if isinstance(response, list):
                response = response[0]

            if response["type"] == "accept":
                return tool_obj.invoke(tool_input, config)
            elif response["type"] == "edit":
                updated_args = response["args"]["args"]
                return tool_obj.invoke(updated_args, config)
            elif response["type"] == "response":
                return response["args"]
            else:
                raise ValueError(
                    f"Unsupported interrupt response type: {response['type']}"
                )

        return tool_with_human_review

    if func is not None:
        return decorator(func)
    else:
        return decorator


def human_in_the_loop_async(
    func: Optional[Callable] = None,
    *,
    interrupt_config: Optional[HumanInterruptConfig] = None,
) -> Union[Callable[[Callable], BaseTool], BaseTool]:
    """
    A decorator that adds human-in-the-loop review support to an asynchronous tool.

    Supports both syntaxes:
        @human_in_the_loop_async
        @human_in_the_loop_async(interrupt_config={...})

    Args:
        func: The function to decorate. **Do not pass this directly.**
        interrupt_config: Configuration for the human interrupt.

    Returns:
        If `func` is provided, returns the decorated BaseTool.
        If `func` is None, returns a decorator that will decorate the target function.
    """

    def _default_config() -> HumanInterruptConfig:
        return {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
            "allow_ignore": True,
        }

    def decorator(target_func: Callable) -> BaseTool:
        """The actual decorator that wraps the target function."""
        if not isinstance(target_func, BaseTool):
            tool_obj = create_tool(target_func)
        else:
            tool_obj = target_func

        final_config = interrupt_config or _default_config()

        @create_tool(
            tool_obj.name,
            description=tool_obj.description,
            args_schema=tool_obj.args_schema,
        )
        async def atool_with_human_review(
            config: RunnableConfig, **tool_input: Any
        ) -> Any:
            request: HumanInterrupt = {
                "action_request": {
                    "action": tool_obj.name,
                    "args": tool_input,
                },
                "config": final_config,
                "description": f"Please review tool call: {tool_obj.name}",
            }

            response = interrupt([request])
            if isinstance(response, list):
                response = response[0]

            if response["type"] == "accept":
                return await tool_obj.ainvoke(tool_input, config)
            elif response["type"] == "edit":
                updated_args = response["args"]["args"]
                return await tool_obj.ainvoke(updated_args, config)
            elif response["type"] == "response":
                return response["args"]
            else:
                raise ValueError(
                    f"Unsupported interrupt response type: {response['type']}"
                )

        return atool_with_human_review

    if func is not None:
        return decorator(func)
    else:
        return decorator
