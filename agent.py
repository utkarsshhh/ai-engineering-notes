import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
MODEL = "gpt-5-mini"

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "tools_used": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["answer", "tools_used", "confidence"],
            "additionalProperties": False,
        },
    },
}

# --- Tools: plain Python functions ---
def calculator(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))  # fine for learning; never eval untrusted input in production
    except Exception as e:
        return f"Error: {e}"

def fetch_url(url: str) -> str:
    try:
        return requests.get(url, timeout=60).text[:3000]  # truncate to protect the context window
    except Exception as e:
        return f"Error: {e}"

def calculate_length(header_string: str):
    try:
        return len(header_string)
    except Exception as e:
        return f"Error: {e}"
    

TOOLS_IMPL = {"calculator": calculator, "fetch_url": fetch_url,"calculate_length":calculate_length}

# --- Tool schemas: what the model sees ---
TOOLS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate a math expression, e.g. '23 * 47 + 12'.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch the raw text content of a web page.",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}},
    {"type": "function", "function": {
            "name": "calculate_length",
            "description": "Calculate the length of a string",
            "parameters": {"type": "object",
                           "properties": {"header_string": {"type": "string"}},
                           "required": ["header_string"]}}},
]

def run_agent(task: str, max_steps: int = 8):
    messages = [
        {"role": "system", "content": "You are an agent. Use tools when needed. Be concise."},
        {"role": "user", "content": task},
    ]
    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, reasoning_effort="low",response_format=RESPONSE_FORMAT,)
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:  # no more tools needed, so this is the final answer
            result = json.loads(msg.content)
            print(f"\n✅ Final answer (step {step + 1}):")
            print(json.dumps(result, indent=2))
            return result

        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments)
                print(f"🔧 Step {step + 1}: {name}({args})")
                if name not in TOOLS_IMPL:
                    result = f"Error: unknown tool '{name}'. Available: {list(TOOLS_IMPL)}"
                else:
                    result = TOOLS_IMPL[name](**args)
            except Exception as e:
                result = f"Error running {name}: {e}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})

    print("⚠️ Hit max steps")

if __name__ == "__main__":
    run_agent("Get the first heading from the page https://www.davidbaldacci.com and then calclate the length of characters present in it and then divide that by 42?")