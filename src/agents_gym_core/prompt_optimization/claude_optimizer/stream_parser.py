"""Parse Claude stream-json output and print human-readable lines to stdout."""

import json
import sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        print(line, flush=True)
        continue
    t = msg.get("type", "")
    if t == "assistant":
        for block in msg.get("message", {}).get("content", []):
            if block.get("type") == "text":
                print(block["text"], flush=True)
            elif block.get("type") == "tool_use":
                name = block.get("name", "?")
                inp = json.dumps(block.get("input", {}), ensure_ascii=False)
                if len(inp) > 200:
                    inp = inp[:200] + "..."
                print(f"  >> [tool] {name}  {inp}", flush=True)
    elif t == "user":  # the CLI sends tool results as user messages of tool_result blocks
        blocks = msg.get("message", {}).get("content", [])
        for block in blocks if isinstance(blocks, list) else []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            content = block.get("content", "")
            if isinstance(content, list):  # content blocks: show the text ones
                content = "\n".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            content = str(content)
            if len(content) > 300:
                content = content[:300] + "..."
            label = "error" if block.get("is_error") else "result"
            print(f"  << [{label}] {content}", flush=True)
    elif t == "result":
        cost = msg.get("total_cost_usd", msg.get("cost_usd", 0))  # older CLIs: cost_usd
        dur = msg.get("duration_ms", 0)
        status = f"Failed: {msg.get('result')}" if msg.get("is_error") else "Done"
        print(f"\n=== {status}. Cost: ${cost:.2f} | Duration: {dur / 1000:.0f}s ===", flush=True)
