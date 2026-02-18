# Model Routing

`ModelRouterMiddleware` is a middleware designed to **dynamically route inputs to the most suitable model**. It utilizes a "router model" to analyze user requests and selects the best model from a predefined list to handle the current task.

### Parameter Description

| Parameter | Description |
|------|------|
| `router_model` | The model used to execute routing decisions.<br><br>**Type**: `str` \| `BaseChatModel`<br>**Required**: Yes |
| `model_list` | List of model configurations.<br><br>**Type**: `list[ModelDict]`<br>**Required**: Yes |
| `router_prompt` | Custom prompt for the router model.<br><br>**Type**: `str`<br>**Required**: No |

#### `model_list` Configuration Details

Each model configuration is a dictionary containing the following fields:

| Field | Description |
|------|------|
| `model_name` | Unique identifier for the model, using the `provider:model-name` format.<br><br>**Type**: `str`<br>**Required**: Yes |
| `model_description` | Brief description of the model's capabilities or applicable scenarios.<br><br>**Type**: `str`<br>**Required**: Yes |
| `tools` | Whitelist of tools available to this model. If not provided, the model defaults to having permission to use all tools.<br><br>**Type**: `list[BaseTool]`<br>**Required**: No |
| `model_kwargs` | Additional parameters for model loading.<br><br>**Type**: `dict`<br>**Required**: No |
| `model_system_prompt` | System-level prompt for the model.<br><br>**Type**: `str`<br>**Required**: No |
| `model_instance` | An instantiated model object.<br><br>**Type**: `BaseChatModel`<br>**Required**: No |


!!! tip "Notes on `model_instance` field"

    - **If provided**: The instance is used directly. `model_name` serves only as an identifier, and `model_kwargs` is ignored. This applies when not using the library's built-in conversation model management features.
    - **If not provided**: The model is loaded using `load_chat_model` based on `model_name` and `model_kwargs`.
    - **Naming Convention**: In either case, it is recommended to use the `provider:model-name` format for `model_name`.

## Usage Example

**Step 1: Define the Model List**

```python
from langchain_dev_utils.agents.middleware.model_router import ModelDict

model_list: list[ModelDict] = [
    {
        "model_name": "vllm:qwen2.5-7b",
        "model_description": "Suitable for general tasks, such as conversation, text generation, etc.",
        "model_kwargs": {
            "temperature": 0.7,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        "model_system_prompt": "You are an assistant skilled in handling general tasks, such as conversation and text generation.",
    },
    {
        "model_name": "vllm:qwen2.5-vl-7b",
        "model_description": "Suitable for visual tasks",
        "tools": [],  # If the model does not need any tools, set this field to an empty list []
    },
    {
        "model_name": "vllm:glm-4.7-flash",
        "model_description": "Suitable for code generation tasks",
        "tools": [run_python_code],  # Only allow the use of the run_python_code tool
    },
    {
        "model_name": "openai:gpt-4o",
        "model_description": "Suitable for complex comprehensive tasks",
        "model_system_prompt": "You are an assistant skilled in handling complex comprehensive tasks",
        "model_instance": ChatOpenAI(
            model="gpt-4o"
        ),  # Pass the instance directly; model_name acts only as an identifier, and model_kwargs is ignored
    },
]
```

**Step 2: Create an Agent and Enable the Middleware**

```python
from langchain_dev_utils.agents.middleware import ModelRouterMiddleware
from langchain_core.messages import HumanMessage

agent = create_agent(
    model="vllm:qwen2.5-7b",  # This model is just a placeholder; it is dynamically replaced by the middleware
    tools=[get_current_time],
    middleware=[
        ModelRouterMiddleware(
            router_model="vllm:qwen2.5-7b",
            model_list=model_list,
        )
    ],
)

# The routing middleware automatically selects the most suitable model based on the input content
response = agent.invoke({"messages": [HumanMessage(content="Help me write a bubble sort code")]})
print(response)
```

!!!tip "The `tools` parameter"
    When using this middleware, the `tools` parameter in `create_agent` is treated as "global supplementary tools". These global tools are only appended to a model's available tool list if the `tools` field for that model in `model_list` is undefined; furthermore, these global tools cannot be included in the `tools` field of models within `model_list`.

With `ModelRouterMiddleware`, you can easily build a multi-model, multi-capability Agent that automatically selects the optimal model based on the task type, improving response quality and efficiency.

!!! note "Parallel Execution"
    The middleware implementation of model routing assigns only one task for execution at a time. If you want to decompose a task into multiple sub-tasks for parallel execution by multiple models, please refer to [Preset StateGraph Builder Functions](../graph.md).