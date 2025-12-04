from typing import Literal, NotRequired, TypedDict, Union

from langchain_core.language_models.chat_models import BaseChatModel

ChatModelType = Union[type[BaseChatModel], Literal["openai-compatible"]]


ToolChoiceType = list[Literal["auto", "none", "required", "specific"]]

ResponseFormatType = list[Literal["json_schema", "json_mode"]]

ReasoningKeepPolicy = Literal["never", "current", "all"]


class CompatibilityOptions(TypedDict):
    supported_tool_choice: NotRequired[ToolChoiceType]
    supported_response_format: NotRequired[ResponseFormatType]
    reasoning_keep_policy: NotRequired[ReasoningKeepPolicy]
    include_usage: NotRequired[bool]
