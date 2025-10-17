from typing import cast

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_tests.integration_tests.chat_models import (
    ChatModelIntegrationTests,
    magic_function,
    _validate_tool_call_message,
)
import pytest

from langchain_dev_utils.chat_models.base import (
    load_chat_model,
    register_model_provider,
)

load_dotenv()

register_model_provider(
    provider_name="zai",
    chat_model="openai-compatible",
    base_url="https://open.bigmodel.cn/api/paas/v4",
)


class TestStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return cast("type[BaseChatModel]", load_chat_model)

    @property
    def chat_model_params(self) -> dict:
        return {"model": "zai:glm-4.5"}

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def has_structured_output(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        return False

    @property
    def supports_json_mode(self) -> bool:
        """(bool) whether the chat model supports JSON mode."""
        return False

    @pytest.mark.xfail(reason="Custom implementation for ZAI model")
    def test_tool_calling(self, model: BaseChatModel) -> None:
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        tool_choice_value = None if not self.has_tool_choice else "any"
        model_with_tools = model.bind_tools(
            [magic_function], tool_choice=tool_choice_value
        )

        # Test invoke
        query = "What is the value of magic_function(3)? Use the tool."
        result = model_with_tools.invoke(query)
        _validate_tool_call_message(result)

        # Test stream
        full: BaseMessage | None = None
        for chunk in model_with_tools.stream(query):
            full = chunk if full is None else full + chunk  # type: ignore[assignment]
        assert isinstance(full, AIMessage)
        _validate_tool_call_message(full)

    @pytest.mark.xfail(reason="Custom implementation for ZAI model")
    async def test_tool_calling_async(self, model: BaseChatModel) -> None:
        if not self.has_tool_calling:
            pytest.skip("Test requires tool calling.")

        tool_choice_value = None if not self.has_tool_choice else "any"
        model_with_tools = model.bind_tools(
            [magic_function], tool_choice=tool_choice_value
        )

        # Test ainvoke
        query = "What is the value of magic_function(3)? Use the tool."
        result = await model_with_tools.ainvoke(query)
        _validate_tool_call_message(result)

        # Test astream
        full: BaseMessage | None = None
        async for chunk in model_with_tools.astream(query):
            full = chunk if full is None else full + chunk  # type: ignore[assignment]
        assert isinstance(full, AIMessage)
        _validate_tool_call_message(full)
