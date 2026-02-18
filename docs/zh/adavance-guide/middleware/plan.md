# 任务规划

`PlanMiddleware` 是一个用于在执行复杂任务前进行结构化分解与过程管理的中间件。

!!! info "补充说明"

    任务规划是一种高效的上下文工程管理策略。在执行任务之前，大模型首先将整体任务拆解为多个有序的子任务，形成任务规划列表（在本库中称为 plan）。随后按顺序执行各子任务，并在每完成一个步骤后动态更新任务状态，直至所有子任务执行完毕。

## 参数说明

| 参数 | 说明 |
|------|------|
| `system_prompt` | 系统提示词，若为 `None` 则使用默认提示词。<br><br>**类型**: `str`<br>**必填**: 否 |
| `custom_plan_tool_descriptions` | 自定义计划相关工具的描述。<br><br>**类型**: `dict`<br>**必填**: 否 |
| `use_read_plan_tool` | 是否启用读计划工具。<br><br>**类型**: `bool`<br>**必填**: 否<br>**默认值**: `True` |

`custom_plan_tool_descriptions` 字典的键可取以下三个值：

| 键 | 说明 |
|------|------|
| `write_plan` | 写计划工具的描述 |
| `finish_sub_plan` | 完成子计划工具的描述 |
| `read_plan` | 读计划工具的描述 |


## 使用示例

```python
from langchain_dev_utils.agents.middleware import PlanMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    middleware=[
        PlanMiddleware(
            custom_plan_tool_descriptions={
                "write_plan": "用于写计划，将任务拆解为多个有序的子任务。",
                "finish_sub_plan": "用于完成子任务，更新子任务状态为已完成。",
                "read_plan": "用于查询当前的任务规划列表。"
            },
            use_read_plan_tool=True,  # 如果不使用读计划工具，可以设置此参数为 False
        )
    ],
)

response = agent.invoke(
    {"messages": [HumanMessage(content="我要去New York玩几天，帮我规划行程")]}
)
print(response)
```

## 工具说明

`PlanMiddleware` 要求必须使用 `write_plan` 和 `finish_sub_plan` 两个工具，而 `read_plan` 工具默认启用；若不需要使用，可将 `use_read_plan_tool` 参数设为 `False`。

## 与官方 To-do list 中间件的对比

本中间件与 LangChain 官方提供的 **To-do list 中间件** 功能定位相似，但在工具设计上存在差异：

| 特性 | 官方 To-do list 中间件 | 本库 PlanMiddleware |
|------|----------------------|---------------------|
| 工具数量 | 1 个（`write_todo`） | 3 个（`write_plan`、`finish_sub_plan`、`read_plan`） |
| 功能定位 | 面向待办清单（todo list） | 专门用于规划列表（plan list） |
| 操作方式 | 添加和修改通过一个工具完成 | 写入、修改、查询分别由不同工具完成 |

无论是 `todo` 还是 `plan`，其本质都是同一个概念。本中间件区别于官方的关键点在于提供了三个专用工具：

- `write_plan`：用于写入计划或更新计划内容
- `finish_sub_plan`：用于在完成某个子任务后更新其状态
- `read_plan`：用于查询计划内容
