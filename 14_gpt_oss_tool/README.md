# GPT-OSS-20B Tool Calling POC

POC to verify whether `gpt-oss-20b` can handle tool calling for git and basic Linux shell commands.

## Setup

### 1. Start vLLM Server

```bash
vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8045 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.3 \
  --trust-remote-code \
  --max-model-len 35096 \
  --api-key 1234 \
  --enable-auto-tool-choice \
  --tool-call-parser openai
```

Key flags for tool calling:
- `--enable-auto-tool-choice` — allows the model to decide when to call tools
- `--tool-call-parser openai` — uses the OpenAI-compatible tool call format

### 2. Install Dependencies

```bash
pip install openai
```

### 3. Run

```bash
python main.py
```

## How It Works

Single tool exposed to the model:

```
shell(cmd, cwd=None)  →  runs any shell command via subprocess
```

The model runs in a multi-turn agentic loop — it can chain multiple tool calls before giving a final answer.

```
user prompt
  └─► model calls shell(cmd)
        └─► result returned to model
              └─► model calls another tool or gives final answer
```

## Demo Prompts

| Prompt | What it tests |
|--------|--------------|
| `git status` | Basic single tool call |
| Last 5 commits + changed files | Command chaining (`git log`, `git diff-stat`) |
| Any uncommitted changes? | Conditional reasoning (call `git diff` or not) |
| List Python files + line counts | Multi-step (`find` + `wc -l`) |

## References

- [vLLM GPT-OSS Recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#tool-use)
