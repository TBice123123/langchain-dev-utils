from typing import Any

from langchain.agents.middleware.summarization import (
    ContextSize,
    DEFAULT_SUMMARY_PROMPT,
    SummarizationMiddleware as _SummarizationMiddleware,
    TokenCounter,
    _DEFAULT_MESSAGES_TO_KEEP,
    _DEFAULT_TRIM_TOKEN_LIMIT,
)
from langchain_core.messages.utils import count_tokens_approximately

from langchain_dev_utils.chat_models.base import load_chat_model


class SummarizationMiddleware(_SummarizationMiddleware):
    """Initialize summarization middleware.

    Args:
        model: The language model to use for generating summaries.
        trigger: One or more thresholds that trigger summarization.

            Provide a single `ContextSize` tuple or a list of tuples, in which case
            summarization runs when any threshold is breached.

            Examples: `("messages", 50)`, `("tokens", 3000)`, `[("fraction", 0.8),
                ("messages", 100)]`.
        keep: Context retention policy applied after summarization.

            Provide a `ContextSize` tuple to specify how much history to preserve.

            Defaults to keeping the most recent 20 messages.

            Examples: `("messages", 20)`, `("tokens", 3000)`, or
                `("fraction", 0.3)`.
        token_counter: Function to count tokens in messages.
        summary_prompt: Prompt template for generating summaries.
        trim_tokens_to_summarize: Maximum tokens to keep when preparing messages for
            the summarization call.

            Pass `None` to skip trimming entirely.

    Examples:
        ```python
        from langchain_dev_utils.agents.middleware import SummarizationMiddleware

        middleware = SummarizationMiddleware(
            model="vllm:qwen3-4b",
            trigger=("tokens", 100),
            keep=("messages", 2),
        )
        ```
    """

    def __init__(
        self,
        model: str,
        *,
        trigger: ContextSize | list[ContextSize] | None = None,
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        **deprecated_kwargs: Any,
    ) -> None:
        chat_model = load_chat_model(model)

        middleware_kwargs = {}
        if trigger is not None:
            middleware_kwargs["trigger"] = trigger
        if keep is not None:
            middleware_kwargs["keep"] = keep
        if token_counter is not None:
            middleware_kwargs["token_counter"] = token_counter
        if summary_prompt is not None:
            middleware_kwargs["summary_prompt"] = summary_prompt
        if trim_tokens_to_summarize is not None:
            middleware_kwargs["trim_tokens_to_summarize"] = trim_tokens_to_summarize

        super().__init__(
            model=chat_model,
            **middleware_kwargs,
        )
