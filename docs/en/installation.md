# Installation

`langchain-dev-utils` can be installed using various package managers. Choose the tool that best fits your workflow.

## Prerequisites

- Python 3.11 or higher
- A Python package manager (recommended: `uv`)
- API Key from any major language model provider

## Installation Methods

`langchain-dev-utils` supports installation with multiple package managers such as `pip`, `poetry`, and `uv`.

To install the basic version of `langchain-dev-utils`:

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
    pip install -U langchain-dev-utils[standard]
    ```

=== "poetry"
    ```bash
    poetry add langchain-dev-utils[standard]
    ```

=== "uv"
    ```bash
    uv add langchain-dev-utils[standard]
    ```

## Dependencies

The package will automatically install the following dependencies:

- `langchain`
- `langgraph` (installed automatically with `langchain`)

For the standard version, the following additional dependencies will be installed:

- `langchain-openai` (for model management)
- `json-repair` (for middleware tool call error fixes)

## Verification

After installation, verify that the package is correctly installed:

```python
import langchain_dev_utils
print(langchain_dev_utils.__version__)
```

## Running Tests

If you want to contribute to the project or run tests:

```bash
git clone https://github.com/TBice123123/langchain-dev-utils.git
cd langchain-dev-utils
uv sync --group tests
uv run pytest .
```

**Note:**

- You need to create a `.env` file and add the relevant `API_KEY` and `API_BASE`.
- All test cases have been verified to pass. If you encounter individual model-related failures during execution, it might be due to model instability. Please try running the tests again.