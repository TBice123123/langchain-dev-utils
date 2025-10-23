from typing import Optional

from langchain.agents.middleware.summarization import (
    SummarizationMiddleware as _SummarizationMiddleware,
    TokenCounter,
)

from langchain_dev_utils.chat_models.base import load_chat_model


class SummarizationMiddleware(_SummarizationMiddleware):
    """Initialize the summarization middleware.

    Args:
        model: The language model to use for generating summaries. Only string identifiers are supported.
        max_tokens_before_summary: Token threshold to trigger summarization.
            If `None`, summarization is disabled.
        messages_to_keep: Number of recent messages to preserve after summarization.
        token_counter: Function to count tokens in messages.
        summary_prompt: Prompt template for generating summaries.
        summary_prefix: Prefix added to system message when including summary.

    Examples:
        ```python
        from langchain_dev_utils.agents.middleware import SummarizationMiddleware

        middleware = SummarizationMiddleware(model="vllm:qwen3-4b", max_tokens_before_summary=100)
        ```
    """

    def __init__(
        self,
        model: str,
        max_tokens_before_summary: Optional[int] = None,
        messages_to_keep: Optional[int] = None,
        token_counter: Optional[TokenCounter] = None,
        summary_prompt: Optional[str] = None,
        summary_prefix: Optional[str] = None,
    ) -> None:
        chat_model = load_chat_model(model)

        middleware_kwargs = {}
        if max_tokens_before_summary is not None:
            middleware_kwargs["max_tokens_before_summary"] = max_tokens_before_summary
        if messages_to_keep is not None:
            middleware_kwargs["messages_to_keep"] = messages_to_keep
        if token_counter is not None:
            middleware_kwargs["token_counter"] = token_counter
        if summary_prompt is not None:
            middleware_kwargs["summary_prompt"] = summary_prompt
        if summary_prefix is not None:
            middleware_kwargs["summary_prefix"] = summary_prefix

        super().__init__(
            model=chat_model,
            **middleware_kwargs,
        )
