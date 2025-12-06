# State Graph Orchestration

## Overview

Provides practical tools for state graph orchestration. Mainly includes the following functions:

- Orchestrating multiple state graphs in a sequential manner to form a sequential workflow.
- Orchestrating multiple state graphs in a parallel manner to form a parallel workflow.

## Sequential Orchestration

Used to build an agent sequential workflow (Sequential Pipeline). This is a work pattern that decomposes complex tasks into a series of continuous, ordered subtasks, and hands them over to different specialized agents for processing in sequence.

Multiple state graphs can be combined in a sequential orchestration manner through `create_sequential_pipeline`.

**Usage Example**:

Developing a software project typically follows a strict linear process:
1. Requirements Analysis: First, the product manager must clarify "what to do" and produce a detailed product requirements document (PRD).
2. Architecture Design: Then, the architect designs "how to do it" based on the PRD, planning the system blueprint and technology selection.
3. Code Writing: Next, development engineers implement the blueprint into specific code according to the architecture design.
4. Testing and Quality Assurance: Finally, testing engineers verify the code to ensure its quality meets requirements.
This process is interconnected and the order cannot be reversed.

For the above four processes, each process has a specialized agent responsible.
1. Product Manager Agent: Receives user's vague requirements and outputs a structured product requirements document (PRD).
2. Architect Agent: Receives the PRD and outputs system architecture diagrams and technical solutions.
3. Development Engineer Agent: Receives the architecture solution and outputs executable source code.
4. Testing Engineer Agent: Receives the source code and outputs test reports and optimization suggestions.

Through the `create_sequential_pipeline` function, these four agents are seamlessly connected to form a highly automated, clearly divided software development pipeline.

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
    return f"Based on the requirements document, system architecture has been designed, including microservice division, data flow diagrams, and technology stack selection."

@tool
def generate_code(architecture: str) -> str:
    """Generate core code based on architecture design"""
    return f"Based on the architecture design, core business code has been generated, including API interfaces, data models, and business logic implementation."

@tool
def create_tests(code: str) -> str:
    """Create test cases for the generated code"""
    return f"Unit tests, integration tests, and end-to-end test cases have been created for the generated code."

# Build an automated software development sequential workflow (pipeline)
graph = create_sequential_pipeline(
    sub_graphs=[
        create_agent(
            model="vllm:qwen3-4b",
            tools=[analyze_requirements],
            system_prompt="You are a product manager responsible for analyzing user requirements and generating detailed product requirements documents.",
            name="requirements_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[design_architecture],
            system_prompt="You are a system architect responsible for designing system architecture based on requirements documents.",
            name="architecture_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[generate_code],
            system_prompt="You are a senior development engineer responsible for generating core code based on architecture design.",
            name="coding_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[create_tests],
            system_prompt="You are a testing engineer responsible for creating comprehensive test cases for the generated code.",
            name="testing_agent",
        ),
    ],
    state_schema=AgentState,
)

response = graph.invoke({"messages": [HumanMessage("Develop an e-commerce website with user registration, product browsing, and shopping cart functionality")]})
print(response)
```

The generated graph is as follows:

![Sequential Pipeline](../../assets/sequential.png)

!!! note "Note"
    For serially combined graphs, LangGraph's StateGraph provides the add_sequence method as a convenient shorthand. This method works best when nodes are functions (rather than subgraphs). If nodes are subgraphs, the code might look like this:

    ```python
    graph = StateGraph(AgentState)
    graph.add_sequence([("graph1", graph1), ("graph2", graph2), ("graph3", graph3)])
    graph.add_edge("__start__", "graph1")
    graph = graph.compile()
    ```

    However, the above writing is still somewhat cumbersome. Therefore, it's more recommended to use the `create_sequential_pipeline` function, which can quickly build a serial execution graph with one line of code, making it more concise and efficient.

## Parallel Orchestration

Used to build an agent parallel workflow (Parallel Pipeline). Its working principle is to combine multiple state graphs in parallel, executing tasks concurrently for each state graph, thereby improving task execution efficiency.

Through the `create_parallel_pipeline` function, multiple state graphs can be combined in a parallel orchestration manner to achieve the effect of parallel task execution.

### Simple Example
**Usage Example**:

In software development, once the system architecture design is completed, different functional modules can often be developed simultaneously by different teams or engineers because they are relatively independent of each other. This is a typical scenario for parallel work.

Suppose we want to develop an e-commerce website, whose core functions can be divided into three independent modules:
1. User Module (registration, login, personal center)
2. Product Module (display, search, classification)
3. Order Module (placing orders, payment, status query)

If developed serially, the time consumption will be the sum of all three. But if developed in parallel, the total time will be approximately equal to the development time of the longest module, greatly improving efficiency.

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

# Build a parallel workflow (pipeline) for frontend module development
graph = create_parallel_pipeline(
    sub_graphs=[
        create_agent(
            model="vllm:qwen3-4b",
            tools=[develop_user_module],
            system_prompt="You are a frontend development engineer responsible for developing user-related modules.",
            name="user_module_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[develop_product_module],
            system_prompt="You are a frontend development engineer responsible for developing product-related modules.",
            name="product_module_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[develop_order_module],
            system_prompt="You are a frontend development engineer responsible for developing order-related modules.",
            name="order_module_agent",
        ),
    ],
    state_schema=AgentState,
)
response = graph.invoke({"messages": [HumanMessage("Parallel development of three core modules of an e-commerce website")]})
print(response)
```

The generated graph is as follows:

![Parallel Pipeline](../../assets/parallel.png)

### Using Branch Function to Specify Subgraphs for Parallel Execution

Sometimes it's necessary to specify which subgraphs to execute in parallel based on conditions. In this case, a branch function can be used.
The branch function needs to return a list of `Send`.

For example, in the above case, suppose the modules to be developed are specified by the user, then only the specified modules will be executed in parallel.

```python
# Build a parallel pipeline (select subgraphs for parallel execution based on conditions)
from langgraph.types import Send

class DevAgentState(AgentState):
    """Development Agent State"""
    selected_modules: list[tuple[str, str]]


# Specify modules selected by the user
select_modules = [("user_module", "Develop user module"), ("product_module", "Develop product module")]

graph = create_parallel_pipeline(
    sub_graphs=[
        create_agent(
            model="vllm:qwen3-4b",
            tools=[develop_user_module],
            system_prompt="You are a frontend development engineer responsible for developing user-related modules.",
            name="user_module_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[develop_product_module],
            system_prompt="You are a frontend development engineer responsible for developing product-related modules.",
            name="product_module_agent",
        ),
        create_agent(
            model="vllm:qwen3-4b",
            tools=[develop_order_module],
            system_prompt="You are a frontend development engineer responsible for developing order-related modules.",
            name="order_module_agent",
        ),
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