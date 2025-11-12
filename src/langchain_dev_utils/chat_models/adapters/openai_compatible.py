from __future__ import annotations
from collections.abc import AsyncIterator, Iterator
from json import JSONDecodeError
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LangSmithParams, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env, secret_from_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai.chat_models._compat import _convert_from_v1_to_chat_completions
from langchain_openai.chat_models.base import BaseChatOpenAI, _convert_message_to_dict
import openai
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    create_model,
    model_validator,
)
from typing_extensions import Self

from ..types import ToolChoiceType

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[dict[str, Any], type[_BM], type]
_DictOrPydantic = Union[dict, _BM]


class _ModelProviderConfigType(BaseModel):
    supported_tool_choice: ToolChoiceType = Field(default=[])
    keep_reasoning_content: bool = Field(default=False)
    support_json_mode: bool = Field(default=False)


class _BaseChatOpenAICompatible(BaseChatOpenAI):
    """
    Base template class for OpenAI-compatible chat model implementations.

    This class provides a foundation for integrating various LLM providers that offer
    OpenAI-compatible APIs (such as vLLM, OpenRouter, ZAI, Moonshot, and many others).
    It enhances the base OpenAI functionality by:
    1. Supporting `reasoning_content` generation and parsing.
    2. Modifying the default implementation method for structured outputs to ensure broader compatibility.
    3. Improving error messages when API responses are invalid.

    Built on top of `langchain-openai`'s `BaseChatOpenAI`, this template class extends
    capabilities to better support diverse OpenAI-compatible model providers while
    maintaining full compatibility with LangChain's chat model interface.

    Note: This is a template class and should not be exported or instantiated directly.
    Instead, use it as a base class and provide the specific provider name through
    inheritance or the factory function `_create_openai_compatible_model()`.
    """

    model_name: str = Field(alias="model", default="openai compatible model")
    """The name of the model"""
    api_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("OPENAI_COMPATIBLE_API_KEY", default=None),
    )
    """OpenAI Compatible API key"""
    api_base: str = Field(
        default_factory=from_env("OPENAI_COMPATIBLE_API_BASE", default=""),
    )
    """OpenAI Compatible API base URL"""

    model_config = ConfigDict(populate_by_name=True)

    _provider: str = PrivateAttr(default="openai-compatible")

    provider_config: _ModelProviderConfigType = Field(
        default_factory=lambda: _ModelProviderConfigType(),
    )

    @property
    def _supported_tool_choice(self) -> ToolChoiceType:
        return self.provider_config.supported_tool_choice

    @property
    def _keep_reasoning_content(self) -> bool:
        return self.provider_config.keep_reasoning_content

    @property
    def _support_json_mode(self) -> bool:
        return self.provider_config.support_json_mode

    @property
    def _llm_type(self) -> str:
        return f"chat-{self._provider}"

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": f"{self._provider.upper()}_API_KEY"}

    def _get_ls_params(
        self,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = self._provider
        return ls_params

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        messages = self._convert_input(input_).to_messages()
        if stop is not None:
            kwargs["stop"] = stop

        payload = {**self._default_params, **kwargs}

        payload_messages = []

        for m in messages:
            if isinstance(m, AIMessage):
                msg_dict = _convert_message_to_dict(
                    _convert_from_v1_to_chat_completions(m)
                )
                if self._keep_reasoning_content and m.additional_kwargs.get(
                    "reasoning_content"
                ):
                    msg_dict["reasoning_content"] = m.additional_kwargs.get(
                        "reasoning_content"
                    )
                payload_messages.append(msg_dict)
            else:
                payload_messages.append(_convert_message_to_dict(m))

        payload["messages"] = payload_messages
        return payload

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if not (self.api_key and self.api_key.get_secret_value()):
            msg = f"{self._provider.upper()}_API_KEY must be set."
            raise ValueError(msg)
        client_params: dict = {
            k: v
            for k, v in {
                "api_key": self.api_key.get_secret_value() if self.api_key else None,
                "base_url": self.api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
            }.items()
            if v is not None
        }

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[dict] = None,
    ) -> ChatResult:
        """Convert API response to LangChain ChatResult with enhanced content processing.

        Extends base implementation to capture and preserve reasoning content from
        model responses, supporting advanced models that provide reasoning chains
        or thought processes alongside regular responses.

        Handles multiple response formats:
        - Standard OpenAI response objects with `reasoning_content` attribute
        - Responses with `model_extra` containing reasoning data
        - Dictionary responses (pass-through to base implementation)

        Args:
            response: Raw API response (OpenAI object or dict)
            generation_info: Additional generation metadata

        Returns:
            ChatResult with enhanced message containing reasoning content when available
        """
        rtn = super()._create_chat_result(response, generation_info)

        if not isinstance(response, openai.BaseModel):
            return rtn

        for generation in rtn.generations:
            if generation.message.response_metadata is None:
                generation.message.response_metadata = {}
            generation.message.response_metadata["model_provider"] = "openai-compatible"

        choices = getattr(response, "choices", None)
        if choices and hasattr(choices[0].message, "reasoning_content"):
            rtn.generations[0].message.additional_kwargs["reasoning_content"] = choices[
                0
            ].message.reasoning_content
        elif choices and hasattr(choices[0].message, "model_extra"):
            model_extra = choices[0].message.model_extra
            if isinstance(model_extra, dict) and (
                reasoning := model_extra.get("reasoning")
            ):
                rtn.generations[0].message.additional_kwargs["reasoning_content"] = (
                    reasoning
                )

        return rtn

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: Optional[dict],
    ) -> Optional[ChatGenerationChunk]:
        """Convert streaming chunk to generation chunk with reasoning content support.

        Processes streaming response chunks to extract reasoning content alongside
        regular message content, enabling real-time streaming of both response
        text and reasoning chains from compatible models.

        Args:
            chunk: Raw streaming chunk from API
            default_chunk_class: Expected chunk type for validation
            base_generation_info: Base metadata for the generation

        Returns:
            ChatGenerationChunk with reasoning content when present in chunk data
        """
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk):
                generation_chunk.message.response_metadata = {
                    **generation_chunk.message.response_metadata,
                    "model_provider": "openai-compatible",
                }
                if (
                    reasoning_content := top.get("delta", {}).get("reasoning_content")
                ) is not None:
                    generation_chunk.message.additional_kwargs["reasoning_content"] = (
                        reasoning_content
                    )
                elif (reasoning := top.get("delta", {}).get("reasoning")) is not None:
                    generation_chunk.message.additional_kwargs["reasoning_content"] = (
                        reasoning
                    )

        return generation_chunk

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        kwargs["stream_options"] = {"include_usage": True}
        try:
            for chunk in super()._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
        except JSONDecodeError as e:
            raise JSONDecodeError(
                f"{self._provider.title()} API returned an invalid response. "
                "Please check the API status and try again.",
                e.doc,
                e.pos,
            ) from e

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kwargs["stream_options"] = {"include_usage": True}
        try:
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
        except JSONDecodeError as e:
            raise JSONDecodeError(
                f"{self._provider.title()} API returned an invalid response. "
                "Please check the API status and try again.",
                e.doc,
                e.pos,
            ) from e

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except JSONDecodeError as e:
            raise JSONDecodeError(
                f"{self._provider.title()} API returned an invalid response. "
                "Please check the API status and try again.",
                e.doc,
                e.pos,
            ) from e

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except JSONDecodeError as e:
            raise JSONDecodeError(
                f"{self._provider.title()} API returned an invalid response. "
                "Please check the API status and try again.",
                e.doc,
                e.pos,
            ) from e

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]

        tool_names = []
        for tool in formatted_tools:
            if "function" in tool:
                tool_names.append(tool["function"]["name"])
            elif "name" in tool:
                tool_names.append(tool["name"])
            else:
                pass

        support_tool_choice = False
        if tool_choice is not None:
            if isinstance(tool_choice, bool):
                tool_choice = "required"
            if isinstance(tool_choice, str):
                if (
                    tool_choice in ["auto", "none", "required"]
                    and tool_choice in self._supported_tool_choice
                ):
                    support_tool_choice = True

                elif "specific" in self._supported_tool_choice:
                    if tool_choice in tool_names:
                        support_tool_choice = True
                        tool_choice = {
                            "type": "function",
                            "function": {"name": tool_choice},
                        }
            tool_choice = tool_choice if support_tool_choice else None
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal[
            "function_calling",
            "json_mode",
            "json_schema",
        ] = "function_calling",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Configure structured output extraction with provider compatibility handling.

        Enables parsing of model outputs into structured formats (Pydantic models
        or dictionaries) while handling provider-specific method compatibility.
        Falls back from json_schema to function_calling for providers that don't
        support the json_schema method.

        Args:
            schema: Output schema (Pydantic model class or dictionary definition)
            method: Extraction method - defaults to function_calling for compatibility
            include_raw: Whether to include raw model response alongside parsed output
            strict: Schema enforcement strictness (provider-dependent)
            **kwargs: Additional structured output parameters

        Returns:
            Runnable configured for structured output extraction
        """
        # Many providers do not support json_schema method, so fallback to function_calling
        if method == "json_schema":
            method = "function_calling"
        if method == "json_mode" and not self._support_json_mode:
            method = "function_calling"

        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            **kwargs,
        )


def _create_openai_compatible_model(
    provider: str, base_url: str
) -> Type[_BaseChatOpenAICompatible]:
    """Factory function for creating provider-specific OpenAI-compatible model classes.

    Dynamically generates model classes for different OpenAI-compatible providers,
    configuring environment variable mappings and default base URLs specific to each provider.

    Args:
        provider: Provider identifier (e.g., `vllm`)
        base_url: Default API base URL for the provider
        tool_choice: List of tool choices for the model (e.g., ["auto", "none", "any", "required", "specific"])
        keep_reasoning_content: Whether to keep reasoning content in the messages

    Returns:
        Configured model class ready for instantiation with provider-specific settings
    """
    return create_model(
        f"Chat{provider.title()}",
        __base__=_BaseChatOpenAICompatible,
        api_base=(
            str,
            Field(
                default_factory=from_env(
                    f"{provider.upper()}_API_BASE", default=base_url
                ),
            ),
        ),
        api_key=(
            str,
            Field(
                default_factory=secret_from_env(
                    f"{provider.upper()}_API_KEY", default=None
                ),
            ),
        ),
        _provider=(
            str,
            PrivateAttr(default=provider),
        ),
    )
