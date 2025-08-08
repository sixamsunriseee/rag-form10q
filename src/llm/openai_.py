from typing import override, Iterable

from openai import AsyncOpenAI

from src.llm.base import BaseLLM
from src.schema import QueryRoute, Chunk


class OpenLLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = AsyncOpenAI(api_key=api_key)


    @override
    async def get_query_route(self, query: str) -> QueryRoute:
        response = await self.client.responses.parse(
            model=self.model_name,
            input=query,
            temperature=0,
            text_format=QueryRoute
        )

        return response.output_parsed


    @override
    async def query(self, instructions: str, chunks: Iterable[Chunk], query: str) -> str:
        structured_input = {
            'chunks': [chunk.model_dump() for chunk in chunks],
            'query': query
        }

        response = await self.client.responses.create(
            model=self.model_name,
            instructions=instructions,
            input=str(structured_input),
            max_output_tokens=256,
            temperature=0
        )

        return response.output_text
