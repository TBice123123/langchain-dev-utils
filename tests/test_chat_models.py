from typing import Any, cast

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests.chat_models import ChatModelIntegrationTests
import pytest

from langchain_dev_utils.chat_models.base import (
    load_chat_model,
    register_model_provider,
)

load_dotenv()

register_model_provider(
    provider_name="zai",
    chat_model="openai-compatible",
)


class TestStandard(ChatModelIntegrationTests):
    @pytest.fixture
    def model(self, request: Any) -> BaseChatModel:
        """Model fixture."""
        extra_init_params = getattr(request, "param", None) or {}
        if extra_init_params.get("output_version") == "v1":
            pytest.skip("Output version v1 is not supported")
        return self.chat_model_class(
            **{
                **self.standard_chat_model_params,
                **self.chat_model_params,
                **extra_init_params,
            },
        )

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return cast("type[BaseChatModel]", load_chat_model)

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "zai:glm-4.5",
            "extra_body": {
                "thinking": {
                    "type": "disabled",
                }
            },
        }

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
