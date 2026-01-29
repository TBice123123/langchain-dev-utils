from typing import Awaitable, Callable, Literal

from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelResponse,
)
from langchain_core.messages import SystemMessage
from langchain_core.prompts.string import get_template_variables

from langchain_dev_utils._utils import _check_pkg_install


class FormatPromptMiddleware(AgentMiddleware):
    """Format the system prompt with variables from state and context.

    This middleware function extracts template variables from the system prompt
    and populates them with values from the agent's state and runtime context.
    Variables are first resolved from the state, then from the context if not found.

    Args:
        template_format: The format of the template. Defaults to "f-string".

    Example:

        # Use format_prompt middleware instance rather than FormatPromptMiddleware class (Recommended)

        ```python
        from langchain_dev_utils.agents.middleware import format_prompt
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from dataclasses import dataclass

        @dataclass
        class Context:
            name: str
            user: str

        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt="You are a helpful assistant. Your name is {name}. Your user is {user}.",
            middleware=[format_prompt],
            context_schema=Context,
        )
        agent.invoke(
            {
                "messages": [HumanMessage(content="Hello")],
            },
            context=Context(name="assistant", user="Tom"),
        )
        ```

        # Use FormatPromptMiddleware class(Use when template_format is jinja2)

        ```python
        from langchain_dev_utils.agents.middleware import FormatPromptMiddleware
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from dataclasses import dataclass

        @dataclass
        class Context:
            name: str
            user: str

        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt="You are a helpful assistant. Your name is {{ name }}. Your user is {{ user }}.",
            middleware=[FormatPromptMiddleware(template_format="jinja2")],
            context_schema=Context,
        )
        agent.invoke(
            {
                "messages": [HumanMessage(content="Hello")],
            },
            context=Context(name="assistant", user="Tom"),
        )
        ```
    """

    def __init__(
        self,
        *,
        template_format: Literal["f-string", "jinja2"] = "f-string",
    ) -> None:
        super().__init__()

        self.template_format = template_format

        if template_format == "jinja2":
            _check_pkg_install("jinja2")

    def _format_prompt(self, request: ModelRequest) -> str:
        """Add the plan system prompt to the system message."""
        system_msg = request.system_message
        if system_msg is None:
            raise ValueError(
                "system_message must be provided,while use format_prompt in middleware."
            )

        system_prompt = "\n".join(
            [content.get("text", "") for content in system_msg.content_blocks]
        )
        variables = get_template_variables(system_prompt, self.template_format)

        format_params = {}

        state = request.state
        for key in variables:
            if var := state.get(key, None):
                format_params[key] = var

        other_var_keys = set(variables) - set(format_params.keys())

        if other_var_keys:
            context = request.runtime.context
            if context is not None:
                for key in other_var_keys:
                    if var := getattr(context, key, None):
                        format_params[key] = var

        if self.template_format == "jinja2":
            from jinja2 import Template

            template = Template(system_prompt)
            return template.render(**format_params)
        else:
            return system_prompt.format(**format_params)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Update the system prompt with variables from state and context."""
        prompt = self._format_prompt(request)
        request = request.override(system_message=SystemMessage(content=prompt))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Update the system prompt with variables from state and context."""
        prompt = self._format_prompt(request)
        override_request = request.override(
            system_message=SystemMessage(content=prompt)
        )
        return await handler(override_request)


format_prompt = FormatPromptMiddleware()
