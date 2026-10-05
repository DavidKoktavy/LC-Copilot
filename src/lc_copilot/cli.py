"""Ask from the terminal:  python -m lc_copilot.cli "Are partial shipments allowed under LC-2026-0187?" """

import asyncio
import json
import sys

from .agent.graph import Agent
from .services import build_services


async def main(question: str) -> None:
    agent = Agent(await build_services())
    r = await agent.ask(question)
    if r["status"] == "pending_approval":
        print(r["approval_request"]["draft"])
        ok = input("\nApprove and queue this notice? [y/N] ").strip().lower() == "y"
        r = await agent.resume(r["thread_id"], ok, approver="cli-user")
    print(f"\n{r['answer']}\n")
    for c in r.get("citations", []):
        print(f"  [{c['ref']}] {c['title']} — {c['section']}  ({c['chunk_id']})")
    if r.get("sql"):
        print(f"\n  SQL: {r['sql']}")
    print("\n" + json.dumps({k: r[k] for k in ("status", "route", "steps", "meta") if k in r}, indent=2))


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "What is the presentation period under LC-2026-0142?"))
