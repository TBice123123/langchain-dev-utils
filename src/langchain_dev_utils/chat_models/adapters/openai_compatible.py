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
    cast,
)

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    LangSmithParams,
    LanguageModelInput,
    ModelProfile,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env, secret_from_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai.chat_models._compat import _convert_from_v1_to_chat_completions
from langchain_openai.chat_models.base import BaseChatOpenAI, _convert_message_to_dict
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

from ..types import (
    CompatibilityOptions,
    ReasoningKeepPolicy,
    ResponseFormatType,
    ToolChoiceType,
)
from .register_profiles import (
    _get_profile_by_provider_and_model,
    _register_profile_with_provider,
)

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[dict[str, Any], type[_BM], type]
_DictOrPydantic = Union[dict, _BM]


def _get_last_human_message_index(messages: list[BaseMessage]) -> int:
    """find the index of the last HumanMessage in the messages list, return -1 if not found."""
    return next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if isinstance(messages[i], HumanMessage)
        ),
        -1,
    )


def _transform_video_block(block: dict[str, Any]) -> dict:
    """Transform video block to video_url block."""
    if "url" in block:
        return {
            "type": "video_url",
            "video_url": {
                "url": block["url"],
            },
        }
    if "base64" in block or block.get("source_type") == "base64":
        if "mime_type" not in block:
            error_message = "mime_type key is required for base64 data."
            raise ValueError(error_message)
        mime_type = block["mime_type"]
        base64_data = block["data"] if "data" in block else block["base64"]
        return {
            "type": "video_url",
            "video_url": {
                "url": f"data:{mime_type};base64,{base64_data}",
            },
        }
    error_message = "Unsupported source type. Only 'url' and 'base64' are supported."
    raise ValueError(error_message)


def _process_video_input(message: BaseMessage):
    """
    Process BaseMessage with video input.

    Args:
        message (BaseMessage): The HumanMessage instance to process.

    Returns:
        None: The method modifies the message in-place.
    """
    if not message.content:
        return message
    content = message.content

    if not isinstance(content, list):
        return message

    formatted_content = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "video":
            formatted_content.append(_transform_video_block(block))
        else:
            formatted_content.append(block)
    message = message.model_copy(update={"content": formatted_content})
    return message


class _BaseChatOpenAICompatible(BaseChatOpenAI):
    """
    Base template class for OpenAI-compatible chat model implementations.

    This class provides a foundation for integrating various LLM providers that
    offer OpenAI-compatible APIs. It enhances the base OpenAI functionality by:

    **1. Supports output of more types of reasoning content (reasoning_content)**
    ChatOpenAI can only output reasoning content natively supported by official
    OpenAI models, while OpenAICompatibleChatModel can output reasoning content
    from other model providers.

    **2. Dynamically adapts to choose the most suitable structured-output method**
    OpenAICompatibleChatModel selects the best structured-output method (function_calling or json_schema)
    based on the actual capabilities of the model provider.

    **3. Supports configuration of related parameters**
    For cases where parameters differ from the official OpenAI API, this library
    provides the compatibility_options parameter to address this issue. For
    example, when different model providers have inconsistent support for
    tool_choice, you can adapt by setting supported_tool_choice in
    compatibility_options.

    Built on top of `langchain-openai`'s `BaseChatOpenAI`, this template class
    extends capabilities to better support diverse OpenAI-compatible model
    providers while maintaining full compatibility with LangChain's chat model
    interface.

    Note: This is a template class and should not be exported or instantiated
    directly. Instead, use it as a base class and provide the specific provider
    name through inheritance or the factory function
    `create_openai_compatible_model()`.
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

    """Provider Compatibility Options"""
    supported_tool_choice: ToolChoiceType = Field(default_factory=list)
    """Supported tool choice"""
    supported_response_format: ResponseFormatType = Field(default_factory=list)
    """Supported response format"""
    reasoning_keep_policy: ReasoningKeepPolicy = Field(default="never")
    """How to keep reasoning content in the messages"""
    include_usage: bool = Field(default=True)
    """Whether to include usage information in the output"""

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
        payload = {**self._default_params, **kwargs}

        if self._use_responses_api(payload):
            return super()._get_request_payload(input_, stop=stop, **kwargs)

        messages = self._convert_input(input_).to_messages()
        if stop is not None:
            kwargs["stop"] = stop

        payload_messages = []
        last_human_index = -1
        if self.reasoning_keep_policy == "current":
            last_human_index = _get_last_human_message_index(messages)

        for index, m in enumerate(messages):
            if isinstance(m, AIMessage):
                msg_dict = _convert_message_to_dict(
                    _convert_from_v1_to_chat_completions(m)
                )
                if self.reasoning_keep_policy == "all" and m.additional_kwargs.get(
                    "reasoning_content"
                ):
                    msg_dict["reasoning_content"] = m.additional_kwargs.get(
                        "reasoning_content"
                    )
                elif (
                    self.reasoning_keep_policy == "current"
                    and index > last_human_index
                    and m.additional_kwargs.get("reasoning_content")
                ):
                    msg_dict["reasoning_content"] = m.additional_kwargs.get(
                        "reasoning_content"
                    )
                payload_messages.append(msg_dict)
            else:
                if (
                    isinstance(m, HumanMessage) or isinstance(m, ToolMessage)
                ) and isinstance(m.content, list):
                    m = _process_video_input(m)
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

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile if not overridden."""
        if self.profile is None:
            self.profile = cast(
                ModelProfile,
                _get_profile_by_provider_and_model(self._provider, self.model_name),
            )
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
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            for chunk in super()._stream_responses(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
        else:
            if self.include_usage:
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
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            async for chunk in super()._astream_responses(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
        else:
            if self.include_usage:
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
                    and tool_choice in self.supported_tool_choice
                ):
                    support_tool_choice = True

                elif "specific" in self.supported_tool_choice:
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
        ] = "json_schema",
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
            method: Extraction method - defaults to json_schema, it the provider doesn't support json_schema, it will fallback to function_calling
            include_raw: Whether to include raw model response alongside parsed output
            strict: Schema enforcement strictness (provider-dependent)
            **kwargs: Additional structured output parameters

        Returns:
            Runnable configured for structured output extraction
        """
        if method not in ["function_calling", "json_mode", "json_schema"]:
            raise ValueError(
                f"Unsupported method: {method}. Please choose from 'function_calling', 'json_mode', 'json_schema'."
            )
        if (
            method == "json_schema"
            and "json_schema" not in self.supported_response_format
        ):
            method = "function_calling"
        elif (
            method == "json_mode" and "json_mode" not in self.supported_response_format
        ):
            method = "function_calling"

        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            **kwargs,
        )


def _create_openai_compatible_model(
    provider: str,
    base_url: str,
    compatibility_options: Optional[CompatibilityOptions] = None,
    profiles: Optional[dict[str, dict[str, Any]]] = None,
    chat_model_cls_name: Optional[str] = None,
) -> Type[_BaseChatOpenAICompatible]:
    """Factory function for creating provider-specific OpenAI-compatible model classes.

    Dynamically generates model classes for different OpenAI-compatible providers,
    configuring environment variable mappings and default base URLs specific to each provider.

    Args:
        provider: Provider identifier (e.g.`vllm`)
        base_url: Default API base URL for the provider
        compatibility_options: Optional configuration for the provider
        profiles: Optional profiles for the provider
        chat_model_cls_name: Optional name for the model class

    Returns:
        Configured model class ready for instantiation with provider-specific settings
    """
    chat_model_cls_name = chat_model_cls_name or f"Chat{provider.title()}"
    if compatibility_options is None:
        compatibility_options = {}

    if profiles is not None:
        _register_profile_with_provider(provider, profiles)

    return create_model(
        chat_model_cls_name,
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
        supported_tool_choice=(
            ToolChoiceType,
            Field(default=compatibility_options.get("supported_tool_choice", ["auto"])),
        ),
        reasoning_keep_policy=(
            ReasoningKeepPolicy,
            Field(default=compatibility_options.get("reasoning_keep_policy", "never")),
        ),
        supported_response_format=(
            ResponseFormatType,
            Field(default=compatibility_options.get("supported_response_format", [])),
        ),
        include_usage=(
            bool,
            Field(default=compatibility_options.get("include_usage", True)),
        ),
    )
