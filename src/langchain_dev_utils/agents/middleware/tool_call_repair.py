from typing import Any, Awaitable, Callable, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.messages import AIMessage, BaseMessage

from langchain_dev_utils._utils import _check_pkg_install


class ToolCallRepairMiddleware(AgentMiddleware):
    """Middleware to repair invalid tool calls in AIMessages.

    This middleware attempts to repair JSON-formatted tool arguments in
    AIMessages that have invalid tool calls. It uses the `json_repair`
    package to fix common JSON errors.

    Example:
        ```python
        from langchain_dev_utils.agents.middleware import ToolCallRepairMiddleware

        middleware = ToolCallRepairMiddleware()
        ```
    """

    def _repair_msgs(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        _check_pkg_install("json_repair")
        from json import JSONDecodeError

        from json_repair import loads

        results = []
        for msg in messages:
            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "invalid_tool_calls")
                and len(msg.invalid_tool_calls) > 0
            ):
                new_invalid_toolcalls = []
                new_tool_calls = [*msg.tool_calls]

                for invalid_tool_call in msg.invalid_tool_calls:
                    args = invalid_tool_call.get("args")
                    if args:
                        try:
                            args = cast(dict[str, Any], loads(args))
                            new_tool_calls.append(
                                {
                                    "name": invalid_tool_call.get(
                                        "name",
                                    )
                                    or "",
                                    "id": invalid_tool_call.get("id", ""),
                                    "type": "tool_call",
                                    "args": args,
                                }
                            )
                        except JSONDecodeError:
                            new_invalid_toolcalls.append(invalid_tool_call)
                    else:
                        new_invalid_toolcalls.append(invalid_tool_call)

                new_msg = msg.model_copy(
                    update={
                        "tool_calls": new_tool_calls,
                        "invalid_tool_calls": new_invalid_toolcalls,
                    }
                )
                results.append(new_msg)
            else:
                results.append(msg)

        return results

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelCallResult:
        response = handler(request)
        results = self._repair_msgs(response.result)

        return ModelResponse(
            result=results,
            structured_response=response.structured_response,
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        response = await handler(request)
        results = self._repair_msgs(response.result)

        return ModelResponse(
            result=results,
            structured_response=response.structured_response,
        )
