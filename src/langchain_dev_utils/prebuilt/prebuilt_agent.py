from typing import Any, Callable, Literal, Optional, Sequence, Type, Union
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.chat_agent_executor import (
    Prompt,
    StateSchemaType,
    StructuredResponseSchema,
)
from langgraph.pregel.main import RunnableLike
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from ..models.chat_model import load_chat_model


def create_agent(
    model: str,
    tools: Union[Sequence[Union[BaseTool, Callable, dict[str, Any]]], ToolNode],
    *,
    prompt: Optional[Prompt] = None,
    response_format: Optional[
        Union[StructuredResponseSchema, tuple[str, StructuredResponseSchema]]
    ] = None,
    pre_model_hook: Optional[RunnableLike] = None,
    post_model_hook: Optional[RunnableLike] = None,
    state_schema: Optional[StateSchemaType] = None,
    context_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
    version: Literal["v1", "v2"] = "v2",
    name: Optional[str] = None,
    **deprecated_kwargs: Any,
) -> CompiledStateGraph:
    """
    Create a prebuilt agent with string-based model specification.

    This function provides the same functionality as the official `create_react_agent`,
    but with the constraint that the model parameter must be a string that can be
    loaded by the `load_chat_model` function. This allows for more flexible model
    specification using the registered model providers.

    Args:
        model: Model identifier string that can be loaded by `load_chat_model`.
               Can be specified as "provider:model-name" format.
        *: All other parameters are the same as in langgraph.prebuilt.create_react_agent.
           See langgraph.prebuilt.create_react_agent for documentation on available parameters.

    Returns:
        CompiledStateGraph: A compiled state graph representing the agent.

    Raises:
        ImportError: If langgraph.prebuilt.create_react_agent is not available.
        ValueError: If the model string cannot be loaded by load_chat_model.

    Example:
        >>> from langchain_dev_utils import register_model_provider
        >>> from langchain_dev_utils.prebuilt import create_agent
        >>> from langchain_core.tools import tool
        >>> import datetime
        >>>
        >>> # Register a model provider
        >>> register_model_provider(
        ...     provider_name="moonshot",
        ...     chat_model="openai-compatible",
        ...     base_url="https://api.moonshot.cn/v1",
        ... )
        >>>
        >>> @tool
        ... def get_current_time() -> str:
        ...     \"\"\"Get current time.\"\"\"
        ...     return str(datetime.datetime.now().timestamp())
        >>>
        >>> agent = create_agent(
        ...     "moonshot:kimi-k2-0905-preview",
        ...     tools=[get_current_time],
        ...     name="time-agent"
        ... )
        >>> response = agent.invoke({
        ...     "messages": [{"role": "user", "content": "What's the time?"}]
        ... })
        >>> response
    """
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        pass

    chat_model = load_chat_model(model)
    return create_react_agent(
        chat_model,
        tools,
        prompt=prompt,
        response_format=response_format,
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
        state_schema=state_schema,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        version=version,
        name=name,
        **deprecated_kwargs,
    )
