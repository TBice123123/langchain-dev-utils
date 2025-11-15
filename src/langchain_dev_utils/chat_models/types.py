from typing import Literal, Union

from langchain_core.language_models.chat_models import BaseChatModel


ChatModelType = Union[type[BaseChatModel], Literal["openai-compatible"]]


ToolChoiceType = list[Literal["auto", "none", "required", "specific"]]
