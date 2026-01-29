# Installation

`langchain-dev-utils` supports installation via multiple package managers, including `pip`, `poetry`, and `uv`.

To install the base version of `langchain-dev-utils`:

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

To install the full-featured version of `langchain-dev-utils`:

=== "pip"
    ```bash
    pip install -U "langchain-dev-utils[standard]"
    ```

=== "poetry"
    ```bash
    poetry add "langchain-dev-utils[standard]"
    ```

=== "uv"
    ```bash
    uv add langchain-dev-utils[standard]
    ```

## Verifying the Installation

After installation, verify that the package is correctly installed:

```python
import langchain_dev_utils
print(langchain_dev_utils.__version__)
```

## Dependencies

The package automatically installs the following dependencies:

- `langchain`
- `langgraph` (installed alongside `langchain`)

If you install the `standard` version, the following additional dependencies will also be installed:

- `langchain-openai` (for model management)
- `json-repair` (for fixing tool-call errors in middleware)
- `jinja2` (for formatting system prompt templates in middleware)
