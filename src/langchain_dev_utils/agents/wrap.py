from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field


class AgentToolInput(BaseModel):
    request: str = Field(description="The input to the agent")


def wrap_agent_as_tool(
    agent: CompiledStateGraph,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    agent_system_prompt: Optional[str] = None,
) -> BaseTool:
    if agent.name is None:
        raise ValueError("Agent name is must not be None")

    def call_agent(request: str) -> str:
        if agent_system_prompt:
            messages = [
                SystemMessage(content=agent_system_prompt),
                HumanMessage(content=request),
            ]
        else:
            messages = [HumanMessage(content=request)]
        response = agent.invoke({"messages": messages})
        return response["messages"][-1].content

    async def acall_agent(request: str) -> str:
        if agent_system_prompt:
            messages = [
                SystemMessage(content=agent_system_prompt),
                HumanMessage(content=request),
            ]
        else:
            messages = [HumanMessage(content=request)]
        response = await agent.ainvoke({"messages": messages})
        return response["messages"][-1].content

    if tool_name is None:
        tool_name = f"transform_to_{agent.name}"
        if not tool_name.endswith("_agent"):
            tool_name += "_agent"

    if tool_description is None:
        tool_description = f"This tool transforms input to {agent.name}"

    return StructuredTool(
        name=tool_name,
        description=tool_description,
        func=call_agent,
        coroutine=acall_agent,
        args_schema=AgentToolInput,
    )
