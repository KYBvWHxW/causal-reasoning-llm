# Causal Reasoning MCP Server

This project implements a Model Context Protocol (MCP) server for testing causal reasoning capabilities in LLMs, based on the research paper "Causal Reasoning in Large Language Models" (arXiv-2502.10215v1).

## Overview

The server implements causal reasoning tasks using collider graphs, where two independent causes (C1 and C2) can each lead to an effect (E). The implementation supports three domains from the paper:
- Economics
- Meteorology
- Sociology

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure MCP in Windsurf:
Create or edit `~/.codeium/windsurf/mcp_config.json`:
```json
{
  "mcpServers": {
    "causal-reasoning": {
      "command": "python",
      "args": [
        "server.py"
      ],
      "cwd": "/Users/paul/CascadeProjects/causal-reasoning-mcp"
    }
  }
}
```

3. Run the server:
```bash
python server.py
```

## Usage

The server provides a tool called `evaluate_causal_reasoning` that can be used to test causal reasoning scenarios. Parameters:

- `domain`: Domain of the scenario ("economics", "meteorology", or "sociology")
- `c1_state`: State of cause 1 (0 or 1, optional)
- `c2_state`: State of cause 2 (0 or 1, optional)
- `e_state`: State of effect (0 or 1, optional)
- `query_variable`: Variable to query ("C1", "C2", or "E")

## Example

```python
# Example tool call
{
    "domain": "economics",
    "c1_state": 1,
    "c2_state": 0,
    "e_state": 1,
    "query_variable": "C2"
}
```

This will generate a prompt asking about the likelihood of market competition (C2) being present, given that there is an economic recession (C1=1) and a business failure has occurred (E=1).
