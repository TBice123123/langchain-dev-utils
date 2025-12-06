from typing import Any
from langchain_dev_utils.tool_calling import human_in_the_loop_async, InterruptParams
from langgraph.types import interrupt


async def custom_handler(params: InterruptParams) -> Any:
    response = interrupt(
        f"我要调用工具 {params['tool_call_name']}，参数为 {params['tool_call_args']}，请确认是否调用"
    )
    if response["type"] == "accept":
        return await params["tool"].ainvoke(params["tool_call_args"])
    elif response["type"] == "reject":
        return "用户拒绝调用该工具"
    else:
        raise ValueError(f"不支持的响应类型: {response['type']}")


@human_in_the_loop_async(handler=custom_handler)
async def get_weather(city: str) -> str:
    """获取天气信息"""
    return f"{city}天气晴朗"
