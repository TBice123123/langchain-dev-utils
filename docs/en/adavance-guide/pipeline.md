# State Graph Orchestration

## Overview

Provides practical utilities for state graph orchestration. Mainly includes the following features:

- Orchestrating multiple state graphs in sequence to form sequential workflows.
- Orchestrating multiple state graphs in parallel to form parallel workflows.

## Sequential Orchestration

Used for building agent sequential pipelines. This is a work pattern that decomposes complex tasks into a series of continuous, ordered sub-tasks, handled by different specialized agents in sequence.

Multiple state graphs can be combined through sequential orchestration using `create_sequential_pipeline`.

**Usage Example**:

Developing a software project typically follows a strict linear process:

1. Requirements Analysis: First, the product manager must clarify "what to do" and produce a detailed Product Requirements Document (PRD).

2. Architecture Design: Then, the architect designs "how to do it" based on the PRD, planning the system blueprint and technology selection.

3. Code Writing: Next, the development engineer implements the blueprint into specific code according to the architecture design.

4. Testing and Quality Assurance: Finally, the testing engineer verifies the code to ensure its quality meets requirements.

This process is interconnected and the order cannot be reversed.

For the above four processes, each process is handled by a specialized agent:

1. Product Manager Agent: Receives vague user requirements and outputs a structured Product Requirements Document (PRD).

2. Architect Agent: Receives the PRD and outputs system architecture diagrams and technical solutions.

3. Development Engineer Agent: Receives the architecture solution and outputs executable source code.

4. Testing Engineer Agent: Receives the source code and outputs test reports and optimization suggestions.

Through the `create_sequential_pipeline` function, these four agents can be seamlessly connected to form a highly automated software development pipeline with clear responsibilities.

```python
from langchain.agents import AgentState
from langchain_core.messages import HumanMessage
from langchain_dev_utils.agents import create_agent
from langchain_dev_utils.pipeline import create_sequential_pipeline
from langchain_core.tools import tool
from langchain_dev_utils.chat_models import register_model_provider

register_model_provider(
    provider_name="vllm",
    chat_model="openai-compatible",
    base_url="http://localhost:8000/v1",
)


@tool
def analyze_requirements(user_request: str) -> str:
    """Analyze user requirements and generate detailed product requirements document"""
    return f"Based on user request '{user_request}', a detailed product requirements document has been generated, including feature list, user stories, and acceptance criteria."


@tool
def design_architecture(requirements: str) -> str:
    """Design system architecture based on requirements document"""
    return "Based on the requirements document, system architecture has been designed, including microservice division, data flow diagram, and technology stack selection."


@tool
def generate_code(architecture: str) -> str:
    """Generate core code based on architecture design"""
    return "Based on the architecture design, core business code has been generated, including API interfaces, data models, and business logic implementation."


@tool
def create_tests(code: str) -> str:
    """Create test cases for the generated code"""
    return "Unit tests, integration tests, and end-to-end test cases have been created for the generated code."


# Product Manager Agent
requirements_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[analyze_requirements],
    system_prompt="You are a product manager responsible for analyzing user requirements and generating detailed product requirements documents.",
    name="requirements_agent",
)

# System Architect Agent
architecture_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[design_architecture],
    system_prompt="You are a system architect responsible for designing system architecture based on requirements documents.",
    name="architecture_agent",
)

# Senior Development Engineer Agent
coding_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[generate_code],
    system_prompt="You are a senior development engineer responsible for generating core code based on architecture design.",
    name="coding_agent",
)

# Testing Engineer Agent
testing_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[create_tests],
    system_prompt="You are a testing engineer responsible for creating comprehensive test cases for the generated code.",
    name="testing_agent",
)

# Build automated software development sequential workflow (pipeline)
graph = create_sequential_pipeline(
    sub_graphs=[
        requirements_agent,
        architecture_agent,
        coding_agent,
        testing_agent,
    ],
    state_schema=AgentState,
)

response = graph.invoke(
    {"messages": [HumanMessage("Develop an e-commerce website with user registration, product browsing, and shopping cart functionality")]}
)
print(response)
```

The generated graph is as follows:

![Sequential Pipeline](../../assets/sequential.png)

The above example is for reference only. In practice, this example passes the complete context of all previous agents to the current agent in sequence, which may lead to context expansion, affecting performance and effectiveness. It is recommended to adopt either of the following solutions to simplify the context:

1. Use `create_agent` with `middleware` to extract and pass only necessary information;

2. Completely customize the state graph based on `LangGraph` to explicitly control state fields and message flow.

??? example "Reference code for solving with middleware"

    ```python
    from typing import Any

    from langchain.agents import AgentState
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import HumanMessage, RemoveMessage
    from langgraph.runtime import Runtime

    from langchain_dev_utils.agents import create_agent
    from langchain_dev_utils.agents.middleware import format_prompt
    from langchain_dev_utils.pipeline import create_sequential_pipeline


    class DeveloperState(AgentState, total=False):
        requirement: str
        architecture: str
        code: str
        tests: str

    class ClearAgentContextMiddleware(AgentMiddleware):
        state_schema = DeveloperState

        def __init__(self, result_save_key: str) -> None:
            super().__init__()
            self.result_save_key = result_save_key

        def after_agent(
            self, state: DeveloperState, runtime: Runtime
        ) -> dict[str, Any] | None:
            final_message = state["messages"][-1]
            update_key = self.result_save_key
            return {
                "messages": [
                    RemoveMessage(id=msg.id or "") for msg in state["messages"][1:]
                ],
                update_key: final_message.content,
            }

    # Product Manager Agent
    requirements_agent = create_agent(
        model="vllm:qwen3-4b",
        tools=[analyze_requirements],
        system_prompt="You are a product manager responsible for analyzing user requirements and generating detailed product requirements documents.",
        name="requirements_agent",
        state_schema=DeveloperState,
        middleware=[format_prompt, ClearAgentContextMiddleware("requirement")],
    )

    # System Architect Agent
    architecture_agent = create_agent(
        model="vllm:qwen3-4b",
        tools=[design_architecture],
        system_prompt="You are a system architect responsible for designing system architecture based on requirements documents.",
        name="architecture_agent",
        state_schema=DeveloperState,
        middleware=[format_prompt, ClearAgentContextMiddleware("architecture")],
    )

    # Senior Development Engineer Agent
    coding_agent = create_agent(
        model="vllm:qwen3-4b",
        tools=[generate_code],
        system_prompt="You are a senior development engineer responsible for generating core code based on architecture design.",
        name="coding_agent",
        state_schema=DeveloperState,
        middleware=[format_prompt, ClearAgentContextMiddleware("code")],
    )

    # Testing Engineer Agent
    testing_agent = create_agent(
        model="vllm:qwen3-4b",
        tools=[create_tests],
        system_prompt="You are a testing engineer responsible for creating comprehensive test cases for the generated code.",
        name="testing_agent",
        state_schema=DeveloperState,
        middleware=[format_prompt, ClearAgentContextMiddleware("tests")],
    )

    # Build automated software development sequential workflow (pipeline)
    graph = create_sequential_pipeline(
        sub_graphs=[
            requirements_agent,
            architecture_agent,
            coding_agent,
            testing_agent,
        ],
        state_schema=DeveloperState,
    )

    response = graph.invoke(
        {"messages": [HumanMessage("Develop an e-commerce website with user registration, product browsing, and shopping cart functionality")]}
    )
    print(response)
    ```
    In the optimized code, we added four fields `requirement`, `architecture`, `code`, and `tests` to the agent's State Schema to store the final output results of corresponding agents.
    
    At the same time, we customized a middleware `ClearAgentContextMiddleware` to clear the current execution context after each agent finishes and save the final result (final_message.content) to the corresponding key.
    
    Finally, we use the built-in `format_prompt` middleware to dynamically concatenate the output of preceding agents into the `system_prompt` as needed at runtime.

!!! note "Note"
    For serially combined graphs, langgraph's StateGraph provides the add_sequence method as a convenient shorthand. This method is most suitable when nodes are functions (rather than subgraphs). If nodes are subgraphs, the code might look like:

    ```python
    graph = StateGraph(AgentState)
    graph.add_sequence([("graph1", graph1), ("graph2", graph2), ("graph3", graph3)])
    graph.add_edge("__start__", "graph1")
    graph = graph.compile()
    ```

    However, the above writing is still somewhat cumbersome. Therefore, it is more recommended to use the `create_sequential_pipeline` function, which can quickly build a serial execution graph with one line of code, making it more concise and efficient.

## Parallel Orchestration

Used for building agent parallel workflows. Its working principle is to combine multiple state graphs in parallel, executing tasks concurrently for each state graph, thereby improving task execution efficiency.

Through the `create_parallel_pipeline` function, multiple state graphs can be combined in a parallel orchestration manner to achieve the effect of parallel task execution.

### Simple Example
**Usage Example**:

In software development, when the system architecture design is completed, different functional modules can often be developed simultaneously by different teams or engineers because they are relatively independent of each other. This is a typical scenario for parallel work.

Suppose we want to develop an e-commerce website, whose core functions can be divided into three independent modules:
1. User module (registration, login, personal center)
2. Product module (display, search, classification)
3. Order module (placing orders, payment, status query)

If developed serially, the time required would be the sum of all three. But if developed in parallel, the total time will be approximately equal to the development time of the longest module, greatly improving efficiency.

Through the `create_parallel_pipeline` function, assign a specialized module development agent to each module, allowing them to work in parallel.

```python
from langchain_dev_utils.pipeline import create_parallel_pipeline


@tool
def develop_user_module():
    """Develop user module functionality"""
    return "User module development completed, including registration, login, and personal profile management functions."


@tool
def develop_product_module():
    """Develop product module functionality"""
    return "Product module development completed, including product display, search, and classification functions."


@tool
def develop_order_module():
    """Develop order module functionality"""
    return "Order module development completed, including order placement, payment, and order query functions."


# User Module Development Agent
user_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_user_module],
    system_prompt="You are a frontend development engineer responsible for developing user-related modules.",
    name="user_module_agent",
)

# Product Module Development Agent
product_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_product_module],
    system_prompt="You are a frontend development engineer responsible for developing product-related modules.",
    name="product_module_agent",
)

# Order Module Development Agent
order_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_order_module],
    system_prompt="You are a frontend development engineer responsible for developing order-related modules.",
    name="order_module_agent",
)

# Build frontend module development parallel workflow (pipeline)
graph = create_parallel_pipeline(
    sub_graphs=[
        user_module_agent,
        product_module_agent,
        order_module_agent,
    ],
    state_schema=AgentState,
)
response = graph.invoke({"messages": [HumanMessage("Parallel development of three core modules of an e-commerce website")]})
print(response)
```

The generated graph is as follows:

![Parallel Pipeline](../../assets/parallel.png)

### Using Branch Functions to Specify Parallelly Executed Subgraphs

Sometimes it is necessary to specify which subgraphs to execute in parallel based on conditions. In such cases, a branch function can be used.
The branch function needs to return a list of `Send`.

For example, in the above case, suppose the modules to be developed are specified by the user, then only the specified modules will be executed in parallel.

```python
# Build parallel pipeline (select subgraphs to execute in parallel based on conditions)
from langgraph.types import Send


class DevAgentState(AgentState):
    """Development Agent State"""

    selected_modules: list[tuple[str, str]]


# Specify modules selected by the user
select_modules = [("user_module", "Develop user module"), ("product_module", "Develop product module")]

user_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_user_module],
    system_prompt="You are a frontend development engineer responsible for developing user-related modules.",
    name="user_module_agent",
)

product_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_product_module],
    system_prompt="You are a frontend development engineer responsible for developing product-related modules.",
    name="product_module_agent",
)


order_module_agent = create_agent(
    model="vllm:qwen3-4b",
    tools=[develop_order_module],
    system_prompt="You are a frontend development engineer responsible for developing order-related modules.",
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
        "messages": [HumanMessage("Develop some modules of the e-commerce website")],
        "selected_modules": select_modules,
    }
)
print(response)
```

**Important Notes**

- When the `branches_fn` parameter is not passed, all subgraphs will be executed in parallel.
- When the `branches_fn` parameter is passed, which subgraphs to execute is determined by the return value of this function.