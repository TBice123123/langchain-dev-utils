# Installation

`langchain-dev-utils` can be installed using various package managers. Choose the one that best fits your workflow.

## Prerequisites

- Python 3.11 or higher
- A Python package manager (we recommend `uv`)
- An API key from any large language model provider

## Installation Methods

`langchain-dev-utils` supports installation via multiple package managers, including `pip`, `poetry`, and `uv`.

::: code-group

```sh[pip]
pip install -U langchain-dev-utils
```

```sh[poetry]
poetry add langchain-dev-utils
```

```sh[uv]
uv add langchain-dev-utils
```

:::

The commands above install `langchain-dev-utils` along with its core dependencies. To enable all features, install the full-featured version using the following commands:

::: code-group

```sh[pip]
pip install -U langchain-dev-utils[standard]
```

```sh[poetry]
poetry add langchain-dev-utils[standard]
```

```sh[uv]
uv add langchain-dev-utils[standard]
```

:::

## Dependencies

The package automatically installs the following dependencies:

- `langchain`
- `langgraph` (installed automatically alongside `langchain`)

The `standard` extra additionally installs:

- `langchain-openai` (used for model management)
- `json-repair` (used by middleware to fix malformed tool calls)

## Verification

After installation, verify that the package is correctly installed:

```python
import langchain_dev_utils
print(langchain_dev_utils.__version__)
```

## Running Tests

If you'd like to contribute to the project or run the test suite:

```bash
git clone https://github.com/TBice123123/langchain-dev-utils.git
cd langchain-dev-utils
uv sync --group tests
uv run pytest .
```

**Note:**

- You need to create a `.env` file and include your relevant `API_KEY` and `API_BASE` configurations.
- All test cases currently pass. If you encounter occasional model-related failures, they may be due to model instability—please try re-running the tests.