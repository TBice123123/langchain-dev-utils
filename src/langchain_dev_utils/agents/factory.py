from typing import Any, Sequence
from typing import Callable

from langchain.agents import create_agent as _create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ResponseT,
    _InputAgentState,
    _OutputAgentState,
)
from langchain.agents.structured_output import ResponseFormat
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from langgraph.typing import ContextT

from ..chat_models import load_chat_model


def create_agent(  # noqa: PLR0915
    model: str,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[AgentState[ResponseT], ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph[
    AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]
]:
    """
    Create a prebuilt agent with string-based model specification.

    This function provides the same functionality as the official `create_react_agent`,
    but with the constraint that the model parameter must be a string that can be
    loaded by the `load_chat_model` function. This allows for more flexible model
    specification using the registered model providers.

    Args:
        model: Model identifier string that can be loaded by `load_chat_model`.
               Can be specified as "provider:model-name" format.
        *: All other parameters are the same as in langchain.agents.create_agent.
           See langchain.agents.create_agent for documentation on available parameters.

    Returns:
        CompiledStateGraph: A compiled state graph representing the agent.

    Raises:
        ValueError: If the model string cannot be loaded by load_chat_model.

    Example:
        >>> from langchain_dev_utils.chat_models import register_model_provider
        >>> from langchain_dev_utils.agents import create_agent
        >>> from langchain_core.tools import tool
        >>> import datetime
        >>>
        >>> # Register a model provider
        >>> register_model_provider(
        ...     provider_name="vllm",
        ...     chat_model="openai-compatible",
        ...     base_url="http://localhost:8000/v1",
        ... )
        >>>
        >>> @tool
        ... def get_current_time() -> str:
        ...     \"\"\"Get current time.\"\"\"
        ...     return str(datetime.datetime.now().timestamp())
        >>>
        >>> agent = create_agent(
        ...     "vllm:qwen3-4b",
        ...     tools=[get_current_time],
        ...     name="time-agent"
        ... )
        >>> response = agent.invoke({
        ...     "messages": [{"role": "user", "content": "What's the time?"}]
        ... })
        >>> response
    """
    return _create_agent(
        model=load_chat_model(model),
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        response_format=response_format,
        state_schema=state_schema,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
        cache=cache,
    )
