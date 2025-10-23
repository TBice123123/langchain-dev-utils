from langchain.agents.middleware.model_fallback import (
    ModelFallbackMiddleware as _ModelFallbackMiddleware,
)

from langchain_dev_utils.chat_models.base import load_chat_model


class ModelFallbackMiddleware(_ModelFallbackMiddleware):
    """Automatic fallback to alternative models on errors.

    Retries failed model calls with alternative models in sequence until
    success or all models exhausted. Primary model specified in create_agent().

    Args:
        first_model: The first model to try on error. Must be a string identifier.
        additional_models: Additional models to try in sequence on error.

    Example:
        ```python
        from langchain_dev_utils.agents.middleware.model_fallback import ModelFallbackMiddleware
        from langchain_dev_utils.agents import create_agent

        fallback = ModelFallbackMiddleware(
            "vllm:qwen3-8b", ## Try first on error
            "openrouter:meta-llama/llama-3.3-8b-instruct:free" #Then this
        )

        agent = create_agent(
            model="vllm:qwen3-4b",#Primary model
            middleware=[fallback],
        )

        # If primary fails: tries qwen3-8b, then llama-3.3-8b-instruct
        result = await agent.invoke({"messages": [HumanMessage("Hello")]})
        ```
    """

    def __init__(
        self,
        first_model: str,
        *additional_models: str,
    ) -> None:
        first_chat_model = load_chat_model(first_model)

        additional_chat_models = [load_chat_model(model) for model in additional_models]
        super().__init__(
            first_chat_model,
            *additional_chat_models,
        )
