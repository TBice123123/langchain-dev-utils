from typing import Any, Awaitable, Callable, NotRequired, Optional, cast

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_dev_utils.chat_models import load_chat_model
from langchain_dev_utils.message_convert import format_sequence


class ModelDict(TypedDict):
    model_name: str
    model_description: str
    tools: NotRequired[list[BaseTool | dict[str, Any]]]
    model_kwargs: NotRequired[dict[str, Any]]
    model_system_prompt: NotRequired[str]


class SelectModel(BaseModel):
    """Tool for model selection - Must call this tool to return the finally selected model"""

    model_name: str = Field(
        ...,
        description="Selected model name (must be the full model name, for example, openai:gpt-4o)",
    )


_ROUTER_MODEL_PROMPT = """
# Role Description
You are an intelligent routing model, specializing in analyzing task requirements and matching appropriate AI models.

# Available Model List
{model_card}

# Core Responsibilities
1. **Task Analysis**: Deeply understand the type, complexity, and special needs of the user's task
2. **Model Matching**: Select the most appropriate model based on the task characteristics
3. **Tool Call**: **Must call SelectModel tool** to return the final selection

# ⚠️ Important Instructions
**After completing the analysis, you must immediately call the SelectModel tool to return the model selection result.**
**This is the only way to output, and you are forbidden to return the result in any other form.**

# Selection Standards
- Consider the type of the task (dialogue, inference, creation, analysis, etc.)
- Evaluate the complexity of the task
- Match the professional ability and applicable scenario of the model
- Ensure that the model's ability matches the task requirements to a high degree

# Execution Process
1. Analyze user task requirements
2. Compare the capabilities of available models
3. Call **SelectModel tool** to submit selection
4. Task completion

Strictly adhere to tool call requirements!
"""


class ModelRouterState(AgentState):
    router_model_selection: str


class ModelRouterMiddleware(AgentMiddleware):
    """Model routing middleware that automatically selects the most suitable model based on input content.

    Args:
        router_model: Model identifier used for routing selection, it can be a model name or a BaseChatModel instance
        model_list: List of available routing models, each containing model_name, model_description, tools(Optional), model_kwargs(Optional), model_system_prompt(Optional)
        router_prompt: Routing prompt template, uses default template if None

    Examples:
        ```python
        from langchain_dev_utils.agents.middleware import ModelRouterMiddleware

        model_list = [
            {
                "model_name": "vllm:qwen3-8b",
                "model_description": "Suitable for general conversation and text generation tasks"
            },
            {
                "model_name": "openrouter:qwen/qwen3-vl-32b-instruct",
                "model_description": "For visual tasks",
                "tools": []
            },
            {
                "model_name": "openrouter:qwen/qwen3-coder-plus",
                "model_description": "For code generation tasks",
                "tools": [run_python_code]
            }
        ]

        middleware = ModelRouterMiddleware(
            router_model="vllm:qwen3-4b",
            model_list=model_list
        )
        ```
    """

    state_schema = ModelRouterState

    def __init__(
        self,
        router_model: str | BaseChatModel,
        model_list: list[ModelDict],
        router_prompt: Optional[str] = None,
    ) -> None:
        super().__init__()
        if isinstance(router_model, BaseChatModel):
            self.router_model = router_model.with_structured_output(SelectModel)
        else:
            self.router_model = load_chat_model(router_model).with_structured_output(
                SelectModel
            )
        self.model_list = model_list

        if router_prompt is None:
            router_prompt = _ROUTER_MODEL_PROMPT.format(
                model_card=format_sequence(
                    [
                        f"model_name:\n {model['model_name']}\n model_description:\n {model['model_description']}"
                        for model in model_list
                    ],
                    with_num=True,
                )
            )

        self.router_prompt = router_prompt

    def _select_model(self, messages: list[AnyMessage]):
        response = cast(
            SelectModel,
            self.router_model.invoke(
                [SystemMessage(content=self.router_prompt), *messages]
            ),
        )
        return response.model_name if response is not None else "default-model"

    async def _aselect_model(self, messages: list[AnyMessage]):
        response = cast(
            SelectModel,
            await self.router_model.ainvoke(
                [SystemMessage(content=self.router_prompt), *messages]
            ),
        )
        return response.model_name if response is not None else "default-model"

    def before_agent(
        self, state: ModelRouterState, runtime: Runtime
    ) -> dict[str, Any] | None:
        model_name = self._select_model(state["messages"])
        return {"router_model_selection": model_name}

    async def abefore_agent(
        self, state: ModelRouterState, runtime: Runtime
    ) -> dict[str, Any] | None:
        model_name = await self._aselect_model(state["messages"])
        return {"router_model_selection": model_name}

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelCallResult:
        model_dict = {
            item["model_name"]: {
                "tools": item.get("tools", None),
                "kwargs": item.get("model_kwargs", None),
                "system_prompt": item.get("model_system_prompt", None),
            }
            for item in self.model_list
        }
        select_model_name = request.state.get("router_model_selection", "default-model")

        override_kwargs = {}
        if select_model_name != "default-model":
            if select_model_name in model_dict:
                model_values = model_dict.get(select_model_name, {})
                if model_values["kwargs"] is not None:
                    model = load_chat_model(select_model_name, **model_values["kwargs"])
                else:
                    model = load_chat_model(select_model_name)
                override_kwargs["model"] = model
                if model_values["tools"] is not None:
                    override_kwargs["tools"] = model_values["tools"]
                if model_values["system_prompt"] is not None:
                    override_kwargs["system_message"] = SystemMessage(
                        content=model_values["system_prompt"]
                    )
        return handler(request.override(**override_kwargs))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        model_dict = {
            item["model_name"]: {
                "tools": item.get("tools", None),
                "kwargs": item.get("model_kwargs", None),
                "system_prompt": item.get("model_system_prompt", None),
            }
            for item in self.model_list
        }
        select_model_name = request.state.get("router_model_selection", "default-model")
        override_kwargs = {}
        if select_model_name != "default-model":
            if select_model_name in model_dict:
                model_values = model_dict.get(select_model_name, {})
                if model_values["kwargs"] is not None:
                    model = load_chat_model(select_model_name, **model_values["kwargs"])
                else:
                    model = load_chat_model(select_model_name)
                override_kwargs["model"] = model
                if model_values["tools"] is not None:
                    override_kwargs["tools"] = model_values["tools"]
                if model_values["system_prompt"] is not None:
                    override_kwargs["system_message"] = SystemMessage(
                        content=model_values["system_prompt"]
                    )
        return await handler(request.override(**override_kwargs))
