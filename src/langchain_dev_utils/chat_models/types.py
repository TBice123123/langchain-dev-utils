from typing import Literal, NotRequired, TypedDict, Union

from langchain_core.language_models.chat_models import BaseChatModel

ChatModelType = Union[type[BaseChatModel], Literal["openai-compatible"]]


ToolChoiceType = list[Literal["auto", "none", "required", "specific"]]


class CompatibilityOptions(TypedDict):
    supported_tool_choice: NotRequired[ToolChoiceType]
    reasoning_content_keep_type: NotRequired[Literal["discard", "temp", "retain"]]
    support_json_mode: NotRequired[bool]
    include_usage: NotRequired[bool]
