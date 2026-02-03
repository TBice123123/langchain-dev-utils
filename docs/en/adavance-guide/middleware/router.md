# Model Routing

`ModelRouterMiddleware` is a middleware used for **dynamically routing to the most suitable model based on the input content**. It analyzes user requests through a "router model" and selects the most appropriate model from a predefined list to process the current task.

### Parameter Description

| Parameter | Description |
|-----------|-------------|
| `router_model` | The model used to make routing decisions.<br><br>**Type**: `str` \| `BaseChatModel`<br>**Required**: Yes |
| `model_list` | List of model configurations.<br><br>**Type**: `list[ModelDict]`<br>**Required**: Yes |
| `router_prompt` | Custom prompt for the router model.<br><br>**Type**: `str`<br>**Required**: No |

#### `model_list` Configuration Description

Each model configuration is a dictionary containing the following fields:

| Field | Description |
|-------|-------------|
| `model_name` | Unique identifier for the model, using the `provider:model-name` format.<br><br>**Type**: `str`<br>**Required**: Yes |
| `model_description` | Brief description of the model's capabilities or suitable scenarios.<br><br>**Type**: `str`<br>**Required**: Yes |
| `tools` | Whitelist of tools callable by this model.<br><br>**Type**: `list[BaseTool]`<br>**Required**: No |
| `model_kwargs` | Additional parameters for model loading.<br><br>**Type**: `dict`<br>**Required**: No |
| `model_system_prompt` | System-level prompt for the model.<br><br>**Type**: `str`<br>**Required**: No |
| `model_instance` | Instantiated model object.<br><br>**Type**: `BaseChatModel`<br>**Required**: No |


!!! tip "Notes on the `model_instance` Field"

    - **If provided**: Use this instance directly. `model_name` is used only for identification, and `model_kwargs` is ignored. Suitable for cases where the library's chat model management functions are not used.
    - **If not provided**: Load the model using `load_chat_model` based on `model_name` and `model_kwargs`.
    - **Naming format**: Regardless of the case, it is recommended to use the `provider:model-name` format for `model_name`.


## Usage Example

**Step 1: Define the Model List**

```python
from langchain_dev_utils.agents.middleware.model_router import ModelDict

model_list: list[ModelDict] = [
    {
        "model_name": "vllm:qwen3-8b",
        "model_description": "Suitable for general tasks, such as dialogue, text generation, etc.",
        "model_kwargs": {
            "temperature": 0.7,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        "model_system_prompt": "You are an assistant skilled in handling general tasks, such as dialogue, text generation, etc.",
    },
    {
        "model_name": "vllm:qwen3-vl-2b",
        "model_description": "Suitable for visual tasks",
        "tools": [],  # If the model does not require any tools, set this field to an empty list []
    },
    {
        "model_name": "vllm:qwen3-coder-flash",
        "model_description": "Suitable for code generation tasks",
        "tools": [run_python_code],  # Only allow the run_python_code tool
    },
    {
        "model_name": "openai:gpt-4o",
        "model_description": "Suitable for comprehensive and high-difficulty tasks",
        "model_system_prompt": "You are an assistant skilled in handling comprehensive and high-difficulty tasks",
        "model_instance": ChatOpenAI(
            model="gpt-4o"
        ),  # Pass the instance directly. Here, model_name is used only for identification, and model_kwargs is ignored.
    },
]
```

**Step 2: Create an Agent and Enable the Middleware**

```python
from langchain_dev_utils.agents.middleware import ModelRouterMiddleware
from langchain_core.messages import HumanMessage

agent = create_agent(
    model="vllm:qwen3-4b",  # This model serves only as a placeholder; it will be dynamically replaced by the middleware
    tools=[run_python_code, get_current_time],
    middleware=[
        ModelRouterMiddleware(
            router_model="vllm:qwen3-4b",
            model_list=model_list,
        )
    ],
)

# The routing middleware will automatically select the most suitable model based on the input content
response = agent.invoke({"messages": [HumanMessage(content="Help me write a bubble sort code")]})
print(response)
```

With `ModelRouterMiddleware`, you can easily build a multi-model, multi-capability Agent that automatically selects the optimal model based on the task type, improving response quality and efficiency.

!!! note "Parallel Execution"
    The middleware-based model routing implementation assigns only one task for execution at a time. If you want to break down a task into multiple subtasks for parallel execution by multiple models, please refer to [Predefined StateGraph Construction Functions](../graph.md).