from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.prompts.string import get_template_variables


@dynamic_prompt
def format_prompt(request: ModelRequest) -> str:
    """Format the system prompt with variables from state and context.

    This middleware function extracts template variables from the system prompt
    and populates them with values from the agent's state and runtime context.
    Variables are first resolved from the state, then from the context if not found.

    Example:
        >>> from langchain_dev_utils.agents.middleware.format_prompt import format_prompt
        >>> from langchain.agents import create_agent
        >>> from langchain_core.messages import HumanMessage
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        ... class Context:
        ...       name: str
        ...       user: str
        >>>
        >>> agent=create_agent(
        ...     model=model,
        ...     tools=tools,
        ...     system_prompt="You are a helpful assistant. Your name is {name}. Your user is {user}.",
        ...     middleware=[format_prompt],
        ...     context_schema=Context,
        ... )
        >>> agent.invoke(
        ...     {
        ...         "messages": [HumanMessage(content="Hello")],
        ...     },
        ...     context=Context(name="assistant", user="Tom"),
        ... )

    """
    system_msg = request.system_message
    if system_msg is None:
        raise ValueError(
            "system_message must be provided,while use format_prompt in middleware."
        )

    system_prompt = "\n".join(
        [content.get("text", "") for content in system_msg.content_blocks]
    )
    variables = get_template_variables(system_prompt, "f-string")

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

    return system_prompt.format(**format_params)
