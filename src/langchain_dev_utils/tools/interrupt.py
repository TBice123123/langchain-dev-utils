from typing import Any, Callable, Dict, Optional, TypedDict, Union, overload

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from langgraph.prebuilt.interrupt import HumanInterrupt
from langgraph.types import interrupt


class InterruptParams(TypedDict):
    tool_call_name: str
    tool_call_args: Dict[str, Any]
    tool: BaseTool
    config: RunnableConfig


HumanInterruptHandler = Callable[[InterruptParams], Any]


@overload
def human_in_the_loop(
    func: Callable,
) -> BaseTool:
    """
    Decorator for adding human-in-the-loop review to a synchronous tool function.
    
    Usage: @human_in_the_loop
    """
    ...


@overload
def human_in_the_loop(
    *,
    handler: Optional[HumanInterruptHandler] = None,
) -> Callable[[Callable], BaseTool]:
    """
    Decorator for adding human-in-the-loop review to a synchronous tool function with custom handler.
    
    Usage: @human_in_the_loop(handler=custom_handler)
    """
    ...


@overload
def human_in_the_loop_async(
    func: Callable,
) -> BaseTool:
    """
    Decorator for adding human-in-the-loop review to an asynchronous tool function.
    
    Usage: @human_in_the_loop_async
    """
    ...


@overload
def human_in_the_loop_async(
    *,
    handler: Optional[HumanInterruptHandler] = None,
) -> Callable[[Callable], BaseTool]:
    """
    Decorator for adding human-in-the-loop review to an asynchronous tool function with custom handler.
    
    Usage: @human_in_the_loop_async(handler=custom_handler)
    """
    ...


def default_handler(params: InterruptParams) -> Any:
    request: HumanInterrupt = {
        "action_request": {
            "action": params["tool_call_name"],
            "args": params["tool_call_args"],
        },
        "config": {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
            "allow_ignore": True,
        },
        "description": f"Please review tool call: {params['tool_call_name']}",
    }
    response = interrupt([request])[0]

    if response["type"] == "accept":
        return params["tool"].invoke(params["tool_call_args"], params["config"])
    elif response["type"] == "edit":
        updated_args = response["args"]["args"]
        return params["tool"].invoke(updated_args, params["config"])
    elif response["type"] == "response":
        return response["args"]
    else:
        raise ValueError(f"Unsupported interrupt response type: {response['type']}")


async def default_handler_async(params: InterruptParams) -> Any:
    request: HumanInterrupt = {
        "action_request": {
            "action": params["tool_call_name"],
            "args": params["tool_call_args"],
        },
        "config": {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
            "allow_ignore": True,
        },
        "description": f"Please review tool call: {params['tool_call_name']}",
    }
    response = interrupt([request])[0]

    if response["type"] == "accept":
        return await params["tool"].ainvoke(params["tool_call_args"], params["config"])
    elif response["type"] == "edit":
        updated_args = response["args"]["args"]
        return await params["tool"].ainvoke(updated_args, params["config"])
    elif response["type"] == "response":
        return response["args"]
    else:
        raise ValueError(f"Unsupported interrupt response type: {response['type']}")


def human_in_the_loop(
    func: Optional[Callable] = None,
    *,
    handler: Optional[HumanInterruptHandler] = None,
) -> Union[Callable[[Callable], BaseTool], BaseTool]:
    """
    A decorator that adds human-in-the-loop review support to a synchronous tool.

    Supports both syntaxes:
        @human_in_the_loop
        @human_in_the_loop(handler=fn)

    Args:
        func: The function to decorate. **Do not pass this directly.**
        handler: Configuration for the human interrupt.

    Returns:
        If `func` is provided, returns the decorated BaseTool.
        If `func` is None, returns a decorator that will decorate the target function.
    """

    def decorator(target_func: Callable) -> BaseTool:
        """The actual decorator that wraps the target function."""
        if not isinstance(target_func, BaseTool):
            tool_obj = create_tool(target_func)
        else:
            tool_obj = target_func

        handler_func: HumanInterruptHandler = handler or default_handler

        @create_tool(
            tool_obj.name,
            description=tool_obj.description,
            args_schema=tool_obj.args_schema,
        )
        def tool_with_human_review(config: RunnableConfig, **tool_input: Any) -> Any:
            return handler_func(
                {
                    "tool_call_name": tool_obj.name,
                    "tool_call_args": tool_input,
                    "tool": tool_obj,
                    "config": config,
                }
            )

        return tool_with_human_review

    if func is not None:
        return decorator(func)
    else:
        return decorator


def human_in_the_loop_async(
    func: Optional[Callable] = None,
    *,
    handler: Optional[HumanInterruptHandler] = None,
) -> Union[Callable[[Callable], BaseTool], BaseTool]:
    """
    A decorator that adds human-in-the-loop review support to an asynchronous tool.

    Supports both syntaxes:
        @human_in_the_loop_async
        @human_in_the_loop_async(handler=fn)

    Args:
        func: The function to decorate. **Do not pass this directly.**
        handler: Configuration for the human interrupt.

    Returns:
        If `func` is provided, returns the decorated BaseTool.
        If `func` is None, returns a decorator that will decorate the target function.
    """

    def decorator(target_func: Callable) -> BaseTool:
        """The actual decorator that wraps the target function."""
        if not isinstance(target_func, BaseTool):
            tool_obj = create_tool(target_func)
        else:
            tool_obj = target_func

        handler_func: HumanInterruptHandler = handler or default_handler_async

        @create_tool(
            tool_obj.name,
            description=tool_obj.description,
            args_schema=tool_obj.args_schema,
        )
        async def atool_with_human_review(
            config: RunnableConfig,
            **tool_input: Any,
        ) -> Any:
            return await handler_func(
                {
                    "tool_call_name": tool_obj.name,
                    "tool_call_args": tool_input,
                    "tool": tool_obj,
                    "config": config,
                }
            )

        return atool_with_human_review

    if func is not None:
        return decorator(func)
    else:
        return decorator
