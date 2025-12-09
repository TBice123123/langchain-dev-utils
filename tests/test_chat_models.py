import base64
from typing import Any, cast

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_tests.integration_tests.chat_models import ChatModelIntegrationTests

from langchain_dev_utils.chat_models.base import load_chat_model


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


class TestImageProcessing:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.model = load_chat_model("zai:glm-4.6v")
        self.image_url = "https://cloudcovert-1305175928.cos.ap-guangzhou.myqcloud.com/%E5%9B%BE%E7%89%87grounding.PNG"
        self.image_data = base64.b64encode(httpx.get(self.image_url).content).decode(
            "utf-8"
        )

    def test_image_url_format(self) -> None:
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Give a concise description of this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self.image_data}"},
                },
            ],
        )
        _ = self.model.invoke([message])

    def test_image_base64_format(self) -> None:
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Give a concise description of this image."},
                {
                    "type": "image",
                    "base64": self.image_data,
                    "mime_type": "image/png",
                },
            ],
        )
        _ = self.model.invoke([message])

    def test_image_url_direct_format(self) -> None:
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Give a concise description of this image.  answer in Chinese",
                },
                {
                    "type": "image",
                    "url": self.image_url,
                },
            ],
        )
        _ = self.model.invoke([message])

    def test_oai_format_tool_message(self) -> None:
        oai_format_message = ToolMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self.image_data}"},
                },
            ],
            tool_call_id="1",
            name="random_image",
        )

        messages = [
            HumanMessage(
                "get a random diagram using the tool and give it a concise description，answer in Chinese"
            ),
            AIMessage(
                [],
                tool_calls=[
                    {
                        "type": "tool_call",
                        "id": "1",
                        "name": "random_image",
                        "args": {},
                    }
                ],
            ),
            oai_format_message,
        ]

        def random_image() -> str:
            """Return a random image."""
            return ""

        _ = self.model.bind_tools([random_image]).invoke(messages)

    def test_standard_format_tool_message(self) -> None:
        standard_format_message = ToolMessage(
            content=[
                {
                    "type": "image",
                    "base64": self.image_data,
                    "mime_type": "image/png",
                },
            ],
            tool_call_id="1",
            name="random_image",
        )

        messages = [
            HumanMessage(
                "get a random diagram using the tool and give it a concise description，answer in Chinese"
            ),
            AIMessage(
                [],
                tool_calls=[
                    {
                        "type": "tool_call",
                        "id": "1",
                        "name": "random_image",
                        "args": {},
                    }
                ],
            ),
            standard_format_message,
        ]

        def random_image() -> str:
            """Return a random image."""
            return ""

        _ = self.model.bind_tools([random_image]).invoke(messages)
