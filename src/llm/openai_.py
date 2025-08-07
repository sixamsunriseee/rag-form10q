import asyncio
from typing import override

from openai import AsyncOpenAI

from src.llm.base import BaseLLM
from config import OPENAI_MODEL, OPENAI_KEY


class OpenLLM(BaseLLM):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_KEY)

    @override
    async def query(self, instructions: str, query: str) -> str:
        response = await self.client.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            input=query,
            max_output_tokens=256,
        )

        return response.output_text
