# ChatModel Module API Reference

## register_model_provider

Register a provider for chat models.

```python
def register_model_provider(
    provider_name: str,
    chat_model: ChatModelType,
    base_url: Optional[str] = None,
    model_profiles: Optional[dict[str, dict[str, Any]]] = None,
    compatibility_options: Optional[CompatibilityOptions] = None,
) -> None:
```

**Parameter Description:**

- `provider_name`: String type, required, custom provider name
- `chat_model`: ChatModel class or supported provider string type, required
- `base_url`: Optional string type, provider's BaseURL
- `model_profiles`: Optional dictionary type, profiles of models supported by the provider, in format `{model_name: model_profile}`
- `compatibility_options`: Optional CompatibilityOptions type, compatibility options

**Example:**

```python
register_model_provider("fakechat",FakeChatModel)
register_model_provider("vllm", "openai-compatible", base_url="http://localhost:8000/v1")
```

## batch_register_model_provider

Batch register model providers.

```python
def batch_register_model_provider(
    providers: list[ChatModelProvider],
) -> None:
```

**Parameter Description:**

- `providers`: ChatModelProvider list type, required, list of provider configurations

**Example:**

```python
batch_register_model_provider([
    {"provider_name": "fakechat", "chat_model": FakeChatModel},
    {"provider_name": "vllm", "chat_model": "openai-compatible", "base_url": "http://localhost:8000/v1"},
])
```

## load_chat_model

Load a chat model from registered providers.

```python
def load_chat_model(
    model: str,
    *,
    model_provider: Optional[str] = None,
    **kwargs: Any,
) -> BaseChatModel:
```

**Parameter Description:**

- `model`: String type, required, model name, in format `model_name` or `provider_name:model_name`
- `model_provider`: Optional string type, model provider name
- `**kwargs`: Any type, optional, additional model parameters

**Return Value:** BaseChatModel type, loaded chat model instance

**Example:**

```python
model = load_chat_model("vllm:qwen3-4b")
```

## ChatModelType

Types supported for the `chat_model` parameter when registering a model provider.

```python
ChatModelType = Union[type[BaseChatModel], Literal["openai-compatible"]]
```

## ToolChoiceType

Types supported for the `tool_choice` parameter.

```python
ToolChoiceType = list[Literal["auto", "none", "required", "specific"]]
```

## ResponseFormatType

Types supported for `response_format`.
```python
ResponseFormatType = list[Literal["json_schema", "json_mode"]]
```

## ReasoningKeepPolicy

Retention policy for the reasoning_content field in the messages list.

```python
ReasoningKeepPolicy = Literal["never", "current", "all"]
```

## CompatibilityOptions

Compatibility options for model providers.

```python
class CompatibilityOptions(TypedDict):
    supported_tool_choice: NotRequired[ToolChoiceType]
    supported_response_format: NotRequired[ResponseFormatType]
    reasoning_keep_policy: NotRequired[ReasoningKeepPolicy]
    include_usage: NotRequired[bool]
```

**Field Description:**

- `supported_tool_choice`: List of supported `tool_choice` strategies
- `supported_response_format`: List of supported `response_format` methods
- `reasoning_keep_policy`: Retention policy for the `reasoning_content` field in historical messages (messages) passed to the model. Optional values are `never`, `current`, `all`
- `include_usage`: Whether to include `usage` information in the last streaming response result

## ChatModelProvider

Configuration type for chat model providers.

```python
class ChatModelProvider(TypedDict):
    provider_name: str
    chat_model: ChatModelType
    base_url: NotRequired[str]
    model_profiles: NotRequired[dict[str, dict[str, Any]]]
    compatibility_options: NotRequired[CompatibilityOptions]
```

**Field Description:**

- `provider_name`: String type, required, provider name
- `chat_model`: BaseChatModel type or string type, required, supports passing a chat model class or string (currently only supports `openai-compatible`)
- `base_url`: Optional string type, base URL
- `model_profiles`: Optional dictionary type, profiles of models supported by the provider, in format `{model_name: model_profile}`
- `compatibility_options`: Optional CompatibilityOptions type, representing model provider compatibility options