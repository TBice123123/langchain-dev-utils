# ChatModel 模块 API 参考文档

## register_model_provider

注册聊天模型的提供者。

### 函数签名

```python
def register_model_provider(
    provider_name: str,
    chat_model: ChatModelType,
    base_url: Optional[str] = None,
    model_profiles: Optional[dict[str, dict[str, Any]]] = None,
    compatibility_options: Optional[CompatibilityOptions] = None,
) -> None:
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| provider_name | str | 是 | - | 自定义提供商名称 |
| chat_model | ChatModelType | 是 | - | ChatModel 类或支持的提供者字符串类型 |
| base_url | Optional[str] | 否 | None | 提供商的 BaseURL |
| model_profiles | Optional[dict[str, dict[str, Any]]] | 否 | None | 提供商所支持的模型的profile，格式为 `{model_name: model_profile}` |
| compatibility_options | Optional[CompatibilityOptions] | 否 | None | 兼容性选项 |

### 示例

```python
register_model_provider("fakechat",FakeChatModel)
register_model_provider("vllm", "openai-compatible", base_url="http://localhost:8000/v1")
```

---

## batch_register_model_provider

批量注册模型提供者。

### 函数签名

```python
def batch_register_model_provider(
    providers: list[ChatModelProvider],
) -> None:
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| providers | list[ChatModelProvider] | 是 | - | 提供者配置列表 |

### 示例

```python
batch_register_model_provider([
    {"provider_name": "fakechat", "chat_model": FakeChatModel},
    {"provider_name": "vllm", "chat_model": "openai-compatible", "base_url": "http://localhost:8000/v1"},
])
```

---

## load_chat_model

从已注册的提供者加载聊天模型。

### 函数签名

```python
def load_chat_model(
    model: str,
    *,
    model_provider: Optional[str] = None,
    **kwargs: Any,
) -> BaseChatModel:
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| model | str | 是 | - | 模型名称，格式为 `model_name` 或 `provider_name:model_name` |
| model_provider | Optional[str] | 否 | None | 模型提供者名称 |
| **kwargs | Any | 否 | - | 额外的模型参数 |


### 示例

```python
model = load_chat_model("vllm:qwen2.5-7b")
```

---

## create_openai_compatible_model

创建一个 OpenAI 兼容的聊天模型类。

### 函数签名

```python
def create_openai_compatible_model(
    model_provider: str,
    base_url: Optional[str] = None,
    compatibility_options: Optional[CompatibilityOptions] = None,
    model_profiles: Optional[dict[str, dict[str, Any]]] = None,
    chat_model_cls_name: Optional[str] = None,
) -> type[BaseChatModel]:
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| model_provider | str | 是 | - | 模型提供者名称 |
| base_url | Optional[str] | 否 | None | 模型提供者的 BaseURL |
| compatibility_options | Optional[CompatibilityOptions] | 否 | None | 兼容性选项 |
| model_profiles | Optional[dict[str, dict[str, Any]]] | 否 | None | 模型提供者所支持的模型的profile，格式为 `{model_name: model_profile}` |
| chat_model_cls_name | Optional[str] | 否 | None | 自定义的聊天模型类名 |

### 返回值

| 类型 | 描述 |
|------|------|
| type[BaseChatModel] | 动态创建的 OpenAI 兼容聊天模型类 |


### 示例

```python
ChatVLLM = create_openai_compatible_model(
    model_provider="vllm",
    base_url="http://localhost:8000/v1",
    chat_model_cls_name="ChatVLLM",
)
```

---

### ChatModelType

注册模型提供商时`chat_model`参数支持的类型。

### 类型定义

```python
ChatModelType = Union[type[BaseChatModel], Literal["openai-compatible"]]
```

---

## ToolChoiceType

`tool_choice`参数支持的类型。

### 类型定义

```python
ToolChoiceType = list[Literal["auto", "none", "required", "specific"]]
```

---

## ResponseFormatType

`response_format`支持的类型。

### 类型定义

```python
ResponseFormatType = list[Literal["json_schema", "json_mode"]]
```

---

## ReasoningKeepPolicy

messages列表中reasoning_content字段的保留策略。

### 类型定义

```python
ReasoningKeepPolicy = Literal["never", "current", "all"]
```

---

## CompatibilityOptions

模型提供商的兼容性选项。

### 类定义

```python
class CompatibilityOptions(TypedDict, total=False):
    supported_tool_choice: ToolChoiceType
    supported_response_format: ResponseFormatType
    reasoning_keep_policy: ReasoningKeepPolicy
    reasoning_field_name: ReasoningFieldName
    include_usage: bool
```

### 字段说明

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| supported_tool_choice | NotRequired[ToolChoiceType] | 否 | 支持的 `tool_choice` 策略列表 |
| supported_response_format | NotRequired[ResponseFormatType] | 否 | 支持的 `response_format` 方法列表 |
| reasoning_keep_policy | NotRequired[ReasoningKeepPolicy] | 否 | 传给模型的历史消息（messages）中 `reasoning_content` 字段的保留策略。可选值有`never`、`current`、`all` |
| reasoning_field_name | NotRequired[ReasoningFieldName] | 否 | 历史消息（messages）中 `reasoning_content` 字段的名称 |
| include_usage | NotRequired[bool] | 否 | 是否在最后一条流式返回结果中包含 `usage` 信息 |

---

## ChatModelProvider

聊天模型提供者配置类型。

### 类定义

```python
class ChatModelProvider(TypedDict):
    provider_name: str
    chat_model: ChatModelType
    base_url: NotRequired[str]
    model_profiles: NotRequired[dict[str, dict[str, Any]]]
    compatibility_options: NotRequired[CompatibilityOptions]
```

### 字段说明

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| provider_name | str | 是 | 提供者名称 |
| chat_model | ChatModelType | 是 | 支持传入对话模型类或字符串（目前只支持`openai-compatible`） |
| base_url | NotRequired[str] | 否 | 基础 URL |
| model_profiles | NotRequired[dict[str, dict[str, Any]]] | 否 | 提供商所支持的模型的profile，格式为 `{model_name: model_profile}` |
| compatibility_options | NotRequired[CompatibilityOptions] | 否 | 模型提供商兼容性选项 |