# 构建并行管道（根据条件选择并行执行的子图）
from langgraph.types import Send


class DevAgentState(AgentState):
    """开发代理状态"""

    selected_modules: list[tuple[str, str]]


# 指定用户选择的模块
select_modules = [("user_module", "开发用户模块"), ("product_module", "开发商品模块")]

user_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_user_module],
    system_prompt="你是一个前端开发工程师，负责开发用户相关模块。",
    name="user_module_agent",
)

product_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_product_module],
    system_prompt="你是一个前端开发工程师，负责开发商品相关模块。",
    name="product_module_agent",
)


order_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_order_module],
    system_prompt="你是一个前端开发工程师，负责开发订单相关模块。",
    name="order_module_agent",
)


graph = create_parallel_pipeline(
    sub_graphs=[
        user_module_agent,
        product_module_agent,
        order_module_agent,
    ],
    state_schema=DevAgentState,
    branches_fn=lambda state: [
        Send(module_name + "_agent", {"messages": [HumanMessage(task_name)]})
        for module_name, task_name in state["selected_modules"]
    ],
)

response = graph.invoke(
    {
        "messages": [HumanMessage("开发电商网站的部分模块")],
        "selected_modules": select_modules,
    }
)
print(response)
