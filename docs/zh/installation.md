# 安装

`langchain-dev-utils`可以使用各种包管理器安装。选择最适合您工作流程的工具。

## 前置准备

- Python 3.11 或更高版本
- Python 包管理器（推荐使用`uv`）
- 任意的大模型提供商的 `API Key`

## 安装方式

`langchain-dev-utils`支持使用`pip`、`poetry`、`uv`等多种包管理器进行安装。

```bash
# 使用 pip 安装
pip install -U langchain-dev-utils

# 使用 poetry 安装
poetry add langchain-dev-utils

# 使用 uv 安装  
uv add langchain-dev-utils
```

上述会安装`langchain-dev-utils`以及它的基础依赖。如果你想要使用它的完整功能你需要执行下面的命令：

```bash
# 使用 pip 安装 standard 版本
pip install -U langchain-dev-utils[standard]

# 使用 poetry 安装 standard 版本
poetry add langchain-dev-utils[standard]

# 使用 uv 安装 standard 版本  
uv add langchain-dev-utils[standard]
```


## 依赖项

该包会自动安装以下依赖项：

- `langchain`
- `langgraph` (安装`langchain`时会同时也会安装)

如果是 standard 版本，还会安装以下依赖项：

- `langchain-openai`（用于模型管理）
- `json-repair`(用于中间件的工具调用错误修复)

## 验证

安装后，验证包是否正确安装：

```python
import langchain_dev_utils
print(langchain_dev_utils.__version__)
```

## 执行测试

如果您想为项目做贡献或运行测试：

```bash
git clone https://github.com/TBice123123/langchain-dev-utils.git
cd langchain-dev-utils
uv sync --group tests
uv run pytest .
```

**注意：**

- 您需要新建一个 `.env` 文件，并写入相关的`API_KEY`以及`API_BASE`。
- 目前所有测试用例均已通过验证。若运行时出现个别与模型相关的失败，有可能是因为模型不稳定导致的，请尝试重新运行测试。
