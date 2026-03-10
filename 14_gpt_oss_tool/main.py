import json
import subprocess
from openai import OpenAI

BASE_URL = "http://localhost:8045/v1"
API_KEY  = "1234"
MODEL    = "openai/gpt-oss-20b"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Run any shell command (git, ls, cat, pwd, find, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string",  "description": "Shell command to run"},
                "cwd": {"type": "string",  "description": "Working directory (optional)"},
            },
            "required": ["cmd"],
        },
    },
}]


def shell(cmd: str, cwd: str = None) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return (r.stdout + r.stderr).strip() or "(no output)"


def run(prompt: str):
    print(f"\n{'='*60}")
    print(f"USER: {prompt}")
    print('='*60)

    messages = [{"role": "user", "content": prompt}]

    while True:
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg  = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print(f"\nASSISTANT:\n{msg.content}")
            break

        for call in msg.tool_calls:
            args   = json.loads(call.function.arguments)
            result = shell(**args)
            print(f"\n[tool] {args['cmd']}")
            print(f"  {result[:500]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})


if __name__ == "__main__":
    REPO = "/home/anderson/projects/nlp-anderson"

    # --- demo prompts ---
    run(f"What's the git status of {REPO}?")

    run(f"Show me the last 5 commits and which files changed in {REPO}")

    run(f"Any uncommitted changes in {REPO}? If yes, show me the diff.")

    run(f"List all Python files under {REPO}/14_gpt_oss_tool and print their line counts")
