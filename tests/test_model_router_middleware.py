from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage

from langchain_dev_utils.agents import create_agent
from langchain_dev_utils.agents.middleware import ModelRouterMiddleware
from langchain_dev_utils.chat_models import batch_register_model_provider

batch_register_model_provider(
    [
        {
            "provider": "dashscope",
            "chat_model": ChatTongyi,
        },
        {
            "provider": "zai",
            "chat_model": "openai-compatible",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
        },
    ]
)


def test_model_router_middleware():
    agent = create_agent(
        model="dashscope:qwen3-max",
        tools=[],
        middleware=[
            ModelRouterMiddleware(
                router_model="dashscope:qwen-flash",
                model_list=[
                    {
                        "model_name": "dashscope:qwen3-max",
                        "model_description": "最聪明的大模型",
                    },
                    {
                        "model_name": "zai:glm-4.5",
                        "model_description": "编码性能最强的大模型",
                    },
                ],
            )
        ],
    )
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="用python实现一个helloworld,越简单越好，不需要思考"
                )
            ]
        }
    )
    assert response
    assert response["messages"][-1].response_metadata.get("model_name") == "glm-4.5"
