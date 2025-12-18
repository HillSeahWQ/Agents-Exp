import time
import asyncio
from langchain_core.tools import tool

@tool
def sync_tool(x: int) -> str:
    """synchronous tool for testing"""
    t = 4
    time.sleep(t)
    return f"sync result - {t} seconds"

@tool
async def async_tool(y: int) -> str:
    """asynchronous tool for testing"""
    t = 9
    await asyncio.sleep(t)
    return f"async result - {t} seconds"