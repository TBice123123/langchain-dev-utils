# 安装

`langchain-dev-utils`支持使用`pip`、`poetry`、`uv`等多种包管理器进行安装。

安装基础版本的`langchain-dev-utils`：

=== "pip"
    ```bash
    pip install -U langchain-dev-utils
    ```

=== "poetry"
    ```bash
    poetry add langchain-dev-utils
    ```

=== "uv"
    ```bash
    uv add langchain-dev-utils
    ```

安装完整功能版本的`langchain-dev-utils`：


=== "pip"
    ```bash
    pip install -U "langchain-dev-utils[standard]"
    ```

=== "poetry"
    ```bash
    poetry add langchain-dev-utils[standard]
    ```

=== "uv"
    ```bash
    uv add langchain-dev-utils[standard]
    ```

## 验证安装

安装后，验证包是否正确安装：

```python
import langchain_dev_utils
print(langchain_dev_utils.__version__)
```

## 依赖项 

该包会自动安装以下依赖项：

- `langchain`
- `langgraph` (安装`langchain`时会同时也会安装)

如果是 standard 版本，还会安装以下依赖项：

- `langchain-openai`（用于模型管理）
- `json-repair`(用于中间件的工具调用错误修复)


