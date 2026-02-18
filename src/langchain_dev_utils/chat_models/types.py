from typing import Any, Literal, Union

from langchain_core.language_models.chat_models import BaseChatModel
from typing_extensions import NotRequired, TypedDict

ChatModelType = Union[type[BaseChatModel], Literal["openai-compatible"]]


ToolChoiceType = list[Literal["auto", "none", "required", "specific"]]

ResponseFormatType = list[Literal["json_schema", "json_mode"]]

ReasoningKeepPolicy = Literal["never", "current", "all"]

ReasoningFieldName = Literal["reasoning_content", "reasoning"]


class CompatibilityOptions(TypedDict, total=False):
    supported_tool_choice: ToolChoiceType
    supported_response_format: ResponseFormatType
    reasoning_keep_policy: ReasoningKeepPolicy
    reasoning_field_name: ReasoningFieldName
    include_usage: bool


class ChatModelProvider(TypedDict):
    provider_name: str
    chat_model: ChatModelType
    base_url: NotRequired[str]
    model_profiles: NotRequired[dict[str, dict[str, Any]]]
    compatibility_options: NotRequired[CompatibilityOptions]
