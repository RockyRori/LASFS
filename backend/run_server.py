from __future__ import annotations

import asyncio
import sys

import uvicorn


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
