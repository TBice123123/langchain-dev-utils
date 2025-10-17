from langchain_core.tools import tool
from langchain_dev_utils.chat_models import (
    batch_register_model_provider,
    load_chat_model,
)
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_dev_utils.embeddings import load_embeddings, register_embeddings_provider

batch_register_model_provider(
    [
        {
            "provider": "zai",
            "chat_model": "openai-compatible",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
        },
        {
            "provider": "moonshot",
            "chat_model": "openai-compatible",
            "base_url": "https://api.moonshot.cn/v1",
        },
        {
            "provider": "fake",
            "chat_model": FakeChatModel,
        },
    ]
)

register_embeddings_provider(
    "fake",
    FakeEmbeddings,
)


@tool
def get_current_time() -> str:
    """get current time"""
    return "2025-10-17"


# model = load_chat_model("moonshot:kimi-k2-0905-preview").bind_tools([get_current_time])
# response = model.invoke("获取今天的日期")
# print(response)
# print(response.content_blocks)

# model = load_chat_model("fake:fake-model")
# response = model.invoke("hello")
# print(response)
# print(response.content_blocks)

embeddings = load_embeddings("fake:fake-embeddings", size=1024)
print(embeddings.embed_query("hello"))
