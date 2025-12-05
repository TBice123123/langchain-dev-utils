# Installation

`langchain-dev-utils` can be installed using various package managers. Choose the tool that best fits your workflow.

## Prerequisites

- Python 3.11 or higher
- Python package manager (recommended to use `uv`)
- API Key from any large language model provider

## Installation Methods

`langchain-dev-utils` supports installation using various package managers such as `pip`, `poetry`, and `uv`.

```bash
# Install using pip
pip install -U langchain-dev-utils

# Install using poetry
poetry add langchain-dev-utils

# Install using uv  
uv add langchain-dev-utils
```

The above will install `langchain-dev-utils` and its basic dependencies. If you want to use its full functionality, you need to execute the following command:

```bash
# Install standard version using pip
pip install -U langchain-dev-utils[standard]

# Install standard version using poetry
poetry add langchain-dev-utils[standard]

# Install standard version using uv  
uv add langchain-dev-utils[standard]
```


## Dependencies

The package will automatically install the following dependencies:

- `langchain`
- `langgraph` (will be installed when installing `langchain`)

For the standard version, the following dependencies will also be installed:

- `langchain-openai` (for model management)
- `json-repair` (for tool call error fixing in middleware)

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

- You need to create a new `.env` file and write the relevant `API_KEY` and `API_BASE`.
- All test cases have been verified. If individual model-related failures occur during runtime, it might be due to model instability. Please try running the tests again.