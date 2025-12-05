# Tool Calling 模块的 API 参考

## has_tool_calling

检查消息是否包含工具调用。

```python
def has_tool_calling(
    message: AIMessage
) -> bool
```

**参数说明：**

- `message`：AIMessage 类型，必填，待检查的消息

**返回值：** 布尔类型，如果消息包含工具调用返回 True，否则返回 False

**示例：**

```python
if has_tool_calling(response):
    # 处理工具调用
    pass
```

## parse_tool_calling

从消息中解析工具调用参数。

```python
def parse_tool_calling(
    message: AIMessage, first_tool_call_only: bool = False
) -> Union[tuple[str, dict], list[tuple[str, dict]]]]
```

**参数说明：**

- `message`：AIMessage 类型，必填，待解析的消息
- `first_tool_call_only`：布尔类型，可选，是否仅返回第一个工具调用，默认 False

**返回值：** 工具调用名称和参数的元组，或工具调用名称和参数元组的列表

**示例：**

```python
# 获取所有工具调用
tool_calls = parse_tool_calling(response)

# 仅获取第一个工具调用
name, args = parse_tool_calling(response, first_tool_call_only=True)
```

## human_in_the_loop

为**同步工具函数**添加"人在回路"人工审核能力的装饰器。

```python
def human_in_the_loop(
    func: Optional[Callable] = None,
    *,
    handler: Optional[HumanInterruptHandler] = None
) -> Union[Callable[[Callable], BaseTool], BaseTool]
```

**参数说明：**

- `func`：可选可调用类型，待装饰的同步函数（装饰器语法糖）
- `handler`：可选 HumanInterruptHandler 类型，自定义中断处理函数

**返回值：** BaseTool 类型，装饰后的工具实例

**示例：**

```python
@human_in_the_loop
def get_current_time():
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

## human_in_the_loop_async

为**异步工具函数**添加"人在回路"人工审核能力的装饰器。

```python
def human_in_the_loop_async(
    func: Optional[Callable] = None,
    *,
    handler: Optional[HumanInterruptHandler] = None
) -> Union[Callable[[Callable], BaseTool], BaseTool]
```

**参数说明：**

- `func`：可选可调用类型，待装饰的异步函数（装饰器语法糖）
- `handler`：可选 HumanInterruptHandler 类型，自定义中断处理函数

**返回值：** BaseTool 类型，装饰后的异步工具实例

**示例：**

```python
@human_in_the_loop_async
async def get_current_time():
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

## InterruptParams

传递给中断处理函数的参数类型。

```python
class InterruptParams(TypedDict):
    tool_call_name: str
    tool_call_args: Dict[str, Any]
    tool: BaseTool
```

**字段说明：**

- `tool_call_name`：字符串类型，工具调用名称
- `tool_call_args`：字符串到任意值的字典类型，工具调用参数
- `tool`：BaseTool 类型，工具实例

## HumanInterruptHandler

中断处理器函数的类型别名。

```python
HumanInterruptHandler = Callable[[InterruptParams], Any]
```
