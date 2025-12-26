# ChatModel Module API Reference Documentation

## register_model_provider

Register a chat model provider.

### Function Signature

```python
def register_model_provider(
    provider_name: str,
    chat_model: ChatModelType,
    base_url: Optional[str] = None,
    model_profiles: Optional[dict[str, dict[str, Any]]] = None,
    compatibility_options: Optional[CompatibilityOptions] = None,
) -> None:
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| provider_name | str | Yes | - | Custom provider name |
| chat_model | ChatModelType | Yes | - | ChatModel class or supported provider string type |
| base_url | Optional[str] | No | None | Base URL of the provider |
| model_profiles | Optional[dict[str, dict[str, Any]]] | No | None | Profiles of models supported by the provider, format: `{model_name: model_profile}` |
| compatibility_options | Optional[CompatibilityOptions] | No | None | Compatibility options |

### Example

```python
register_model_provider("fakechat", FakeChatModel)
register_model_provider("vllm", "openai-compatible", base_url="http://localhost:8000/v1")
```

---

## batch_register_model_provider

Batch register model providers.

### Function Signature

```python
def batch_register_model_provider(
    providers: list[ChatModelProvider],
) -> None:
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| providers | list[ChatModelProvider] | Yes | - | List of provider configurations |

### Example

```python
batch_register_model_provider([
    {"provider_name": "fakechat", "chat_model": FakeChatModel},
    {"provider_name": "vllm", "chat_model": "openai-compatible", "base_url": "http://localhost:8000/v1"},
])
```

---

## load_chat_model

Load a chat model from registered providers.

### Function Signature

```python
def load_chat_model(
    model: str,
    *,
    model_provider: Optional[str] = None,
    **kwargs: Any,
) -> BaseChatModel:
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| model | str | Yes | - | Model name, format: `model_name` or `provider_name:model_name` |
| model_provider | Optional[str] | No | None | Model provider name |
| **kwargs | Any | No | - | Additional model parameters |

### Example

```python
model = load_chat_model("vllm:qwen3-4b")
```

---

## ChatModelType

Types supported for the `chat_model` parameter when registering a model provider.

### Type Definition

```python
ChatModelType = Union[type[BaseChatModel], Literal["openai-compatible"]]
```

---

## ToolChoiceType

Types supported for the `tool_choice` parameter.

### Type Definition

```python
ToolChoiceType = list[Literal["auto", "none", "required", "specific"]]
```

---

## ResponseFormatType

Types supported for `response_format`.

### Type Definition

```python
ResponseFormatType = list[Literal["json_schema", "json_mode"]]
```

---

## ReasoningKeepPolicy

Retention policy for the reasoning_content field in the messages list.

### Type Definition

```python
ReasoningKeepPolicy = Literal["never", "current", "all"]
```

---

## CompatibilityOptions

Compatibility options for model providers.

### Class Definition

```python
class CompatibilityOptions(TypedDict):
    supported_tool_choice: NotRequired[ToolChoiceType]
    supported_response_format: NotRequired[ResponseFormatType]
    reasoning_keep_policy: NotRequired[ReasoningKeepPolicy]
    include_usage: NotRequired[bool]
```

### Field Description

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| supported_tool_choice | NotRequired[ToolChoiceType] | No | List of supported `tool_choice` strategies |
| supported_response_format | NotRequired[ResponseFormatType] | No | List of supported `response_format` methods |
| reasoning_keep_policy | NotRequired[ReasoningKeepPolicy] | No | Retention policy for the `reasoning_content` field in historical messages (messages) passed to the model. Optional values are `never`, `current`, `all` |
| include_usage | NotRequired[bool] | No | Whether to include `usage` information in the last streaming response result |

---

## ChatModelProvider

Chat model provider configuration type.

### Class Definition

```python
class ChatModelProvider(TypedDict):
    provider_name: str
    chat_model: ChatModelType
    base_url: NotRequired[str]
    model_profiles: NotRequired[dict[str, dict[str, Any]]]
    compatibility_options: NotRequired[CompatibilityOptions]
```

### Field Description

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| provider_name | str | Yes | Provider name |
| chat_model | ChatModelType | Yes | Supports passing a chat model class or string (currently only supports `openai-compatible`) |
| base_url | NotRequired[str] | No | Base URL |
| model_profiles | NotRequired[dict[str, dict[str, Any]]] | No | Profiles of models supported by the provider, format: `{model_name: model_profile}` |
| compatibility_options | NotRequired[CompatibilityOptions] | No | Model provider compatibility options |