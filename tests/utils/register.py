from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_qwq import ChatQwen

from data.alibaba._profiles import _PROFILES as ALI_PROFILES
from data.zhipuai._profiles import _PROFILES as ZAI_PROFILES
from langchain_dev_utils.chat_models import batch_register_model_provider
from langchain_dev_utils.embeddings import batch_register_embeddings_provider


def register_all_model_providers():
    batch_register_model_provider(
        [
            {
                "provider_name": "dashscope",
                "chat_model": ChatQwen,
                "model_profiles": ALI_PROFILES,
            },
            {
                "provider_name": "zai",
                "chat_model": "openai-compatible",
                "model_profiles": ZAI_PROFILES,
            },
        ]
    )


def register_all_embeddings_providers():
    batch_register_embeddings_provider(
        [
            {"provider_name": "dashscope", "embeddings_model": DashScopeEmbeddings},
        ]
    )
