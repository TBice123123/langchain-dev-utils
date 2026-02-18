# ChatModel Module API Reference Documentation

## register_model_provider

Register a provider for chat models.

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
| base_url | Optional[str] | No | None | BaseURL of the provider |
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
model = load_chat_model("vllm:qwen2.5-7b")
```

---

## create_openai_compatible_model

Create an OpenAI compatible chat model class.

### Function Signature

```python
def create_openai_compatible_model(
    model_provider: str,
    base_url: Optional[str] = None,
    compatibility_options: Optional[CompatibilityOptions] = None,
    model_profiles: Optional[dict[str, dict[str, Any]]] = None,
    chat_model_cls_name: Optional[str] = None,
) -> type[BaseChatModel]:
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| model_provider | str | Yes | - | Model provider name |
| base_url | Optional[str] | No | None | BaseURL of the model provider |
| compatibility_options | Optional[CompatibilityOptions] | No | None | Compatibility options |
| model_profiles | Optional[dict[str, dict[str, Any]]] | No | None | Profiles of models supported by the provider, format: `{model_name: model_profile}` |
| chat_model_cls_name | Optional[str] | No | None | Custom chat model class name |

### Return Value

| Type | Description |
|------|-------------|
| type[BaseChatModel] | Dynamically created OpenAI compatible chat model class |

### Example

```python
ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    base_url="http://localhost:8000/v1",
    chat_model_cls_name="ChatVLLM",
)
```

---

### ChatModelType

Types supported by the `chat_model` parameter when registering model providers.

### Type Definition

```python
ChatModelType = Union[type[BaseChatModel], Literal["openai-compatible"]]
```

---

## ToolChoiceType

Types supported by the `tool_choice` parameter.

### Type Definition

```python
ToolChoiceType = list[Literal["auto", "none", "required", "specific"]]
```

---

## ResponseFormatType

Types supported by `response_format`.

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
class CompatibilityOptions(TypedDict, total=False):
    supported_tool_choice: ToolChoiceType
    supported_response_format: ResponseFormatType
    reasoning_keep_policy: ReasoningKeepPolicy
    reasoning_field_name: ReasoningFieldName
    include_usage: bool
```

### Field Description

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| supported_tool_choice | NotRequired[ToolChoiceType] | No | List of supported `tool_choice` strategies |
| supported_response_format | NotRequired[ResponseFormatType] | No | List of supported `response_format` methods |
| reasoning_keep_policy | NotRequired[ReasoningKeepPolicy] | No | Retention policy for the `reasoning_content` field in historical messages (messages) passed to the model. Optional values are `never`, `current`, `all` |
| reasoning_field_name | NotRequired[ReasoningFieldName] | No | Field name for reasoning content in the messages list |
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
| chat_model | ChatModelType | Yes | Support passing chat model class or string (currently only supports `openai-compatible`) |
| base_url | NotRequired[str] | No | Base URL |
| model_profiles | NotRequired[dict[str, dict[str, Any]]] | No | Profiles of models supported by the provider, format: `{model_name: model_profile}` |
| compatibility_options | NotRequired[CompatibilityOptions] | No | Model provider compatibility options |