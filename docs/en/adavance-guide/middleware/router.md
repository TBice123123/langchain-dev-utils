# Model Router

`ModelRouterMiddleware` is a middleware used to **dynamically route to the most suitable model based on input content**. It analyzes user requests using a "router model" and selects the model best suited for the current task from a predefined list.

### Parameter Description

| Parameter | Description |
|------|------|
| `router_model` | The model used to make routing decisions.<br><br>**Type**: `str` \| `BaseChatModel`<br>**Required**: Yes |
| `model_list` | A list of model configurations.<br><br>**Type**: `list[ModelDict]`<br>**Required**: Yes |
| `router_prompt` | Custom prompt for the router model.<br><br>**Type**: `str`<br>**Required**: No |

#### `model_list` Configuration Description

Each model configuration is a dictionary containing the following fields:

| Field | Description |
|------|------|
| `model_name` | The unique identifier for the model, using the `provider:model-name` format.<br><br>**Type**: `str`<br>**Required**: Yes |
| `model_description` | A brief description of the model's capabilities or applicable scenarios.<br><br>**Type**: `str`<br>**Required**: Yes |
| `tools` | A whitelist of tools callable by this model. If not passed, the model defaults to having permission to use all tools.<br><br>**Type**: `list[BaseTool]`<br>**Required**: No |
| `model_kwargs` | Additional parameters when loading the model.<br><br>**Type**: `dict`<br>**Required**: No |
| `model_system_prompt` | System-level prompt for the model.<br><br>**Type**: `str`<br>**Required**: No |
| `model_instance` | An instantiated model object.<br><br>**Type**: `BaseChatModel`<br>**Required**: No |


!!! tip "Explanation of the `model_instance` Field"

    - **If provided**: The instance is used directly, `model_name` serves only as an identifier, and `model_kwargs` is ignored; suitable for scenarios not using this library's chat model management features.
    - **If not provided**: The model is loaded using `load_chat_model` based on `model_name` and `model_kwargs`.
    - **Naming format**: In either case, it is recommended that `model_name` follows the `provider:model-name` format.


## Usage Examples

**Step 1: Define the Model List**

```python
from langchain_dev_utils.agents.middleware.model_router import ModelDict

model_list: list[ModelDict] = [
    {
        "model_name": "vllm:qwen3-8b",
        "model_description": "Suitable for general tasks, such as conversation, text generation, etc.",
        "model_kwargs": {
            "temperature": 0.7,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        "model_system_prompt": "You are an assistant skilled at handling general tasks, such as conversation and text generation.",
    },
    {
        "model_name": "vllm:qwen3-vl-2b",
        "model_description": "Suitable for visual tasks",
        "tools": [],  # If the model does not require any tools, please set this field to an empty list []
    },
    {
        "model_name": "vllm:qwen3-coder-flash",
        "model_description": "Suitable for code generation tasks",
        "tools": [run_python_code],  # Only allow the use of the run_python_code tool
    },
    {
        "model_name": "openai:gpt-4o",
        "model_description": "Suitable for comprehensive and complex tasks",
        "model_system_prompt": "You are an assistant skilled at handling comprehensive high-difficulty tasks",
        "model_instance": ChatOpenAI(
            model="gpt-4o"
        ),  # Pass the instance directly; here model_name serves only as an identifier, model_kwargs is ignored
    },
]
```

**Step 2: Create an Agent and Enable Middleware**

```python
from langchain_dev_utils.agents.middleware import ModelRouterMiddleware
from langchain_core.messages import HumanMessage

agent = create_agent(
    model="vllm:qwen3-4b",  # This model serves as a placeholder; it is dynamically replaced by the middleware
    tools=[get_current_time],
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

!!!tip "tools Parameter"
    After using this middleware, the `tools` parameter of `create_agent` is treated as "global supplementary tools". These global tools are only appended to a model's available tool list if the `tools` field is not defined in that model's `model_list`; furthermore, these global tools must not be included in the `tools` field defined in `model_list`.


With `ModelRouterMiddleware`, you can easily build a multi-model, multi-capability Agent that automatically selects the optimal model based on task type, improving response quality and efficiency.

!!! note "Parallel Execution"
    Implementing model routing via middleware assigns only one task for execution at a time. If you wish to decompose a task into multiple sub-tasks executed in parallel by multiple models, please refer to [Predefined StateGraph Construction Functions](../graph.md).