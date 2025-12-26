# Tool Calling 模块 API 参考文档

## has_tool_calling

检查消息是否包含工具调用。

### 函数签名

```python
def has_tool_calling(
    message: AIMessage
) -> bool
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| message | AIMessage | 是 | - | 待检查的消息 |


### 示例

```python
if has_tool_calling(response):
    # 处理工具调用
    pass
```

---

## parse_tool_calling

从消息中解析工具调用参数。

### 函数签名

```python
def parse_tool_calling(
    message: AIMessage, first_tool_call_only: bool = False
) -> Union[tuple[str, dict], list[tuple[str, dict]]]
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| message | AIMessage | 是 | - | 待解析的消息 |
| first_tool_call_only | bool | 否 | False | 是否仅返回第一个工具调用 |


### 示例

```python
# 获取所有工具调用
tool_calls = parse_tool_calling(response)

# 仅获取第一个工具调用
name, args = parse_tool_calling(response, first_tool_call_only=True)
```

---

## human_in_the_loop

为**同步工具函数**添加"人在回路"人工审核能力的装饰器。

### 函数签名

```python
def human_in_the_loop(
    func: Optional[Callable] = None,
    *,
    handler: Optional[HumanInterruptHandler] = None
) -> Union[Callable[[Callable], BaseTool], BaseTool]
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| func | Optional[Callable] | 否 | None | 待装饰的同步函数（装饰器语法糖） |
| handler | Optional[HumanInterruptHandler] | 否 | None | 自定义中断处理函数 |


### 示例

```python
@human_in_the_loop
def get_current_time():
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

---

## human_in_the_loop_async

为**异步工具函数**添加"人在回路"人工审核能力的装饰器。

### 函数签名

```python
def human_in_the_loop_async(
    func: Optional[Callable] = None,
    *,
    handler: Optional[HumanInterruptHandler] = None
) -> Union[Callable[[Callable], BaseTool], BaseTool]
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| func | Optional[Callable] | 否 | None | 待装饰的异步函数（装饰器语法糖） |
| handler | Optional[HumanInterruptHandler] | 否 | None | 自定义中断处理函数 |


### 示例

```python
@human_in_the_loop_async
async def get_current_time():
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

---

## InterruptParams

传递给中断处理函数的参数类型。

### 类定义

```python
class InterruptParams(TypedDict):
    tool_call_name: str
    tool_call_args: Dict[str, Any]
    tool: BaseTool
```

### 字段说明

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| tool_call_name | str | 是 | 工具调用名称 |
| tool_call_args | Dict[str, Any] | 是 | 工具调用参数 |
| tool | BaseTool | 是 | 工具实例 |

---

## HumanInterruptHandler

中断处理器函数的类型别名。

### 类型定义

```python
HumanInterruptHandler = Callable[[InterruptParams], Any]
```