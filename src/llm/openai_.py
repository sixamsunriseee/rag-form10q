from typing import override

from openai import AsyncOpenAI

from src.llm.base import BaseLLM
from src.llm.instructions import (
    ROUTING_INSTRUCTIONS,
    DECOMPOSITION_INSTRUCTIONS,
    SINGLE_ANSWER_GENERATION_INSTRUCTIONS,
    MULTI_ANSWER_GENERATION_INSTRUCTIONS
)
from src.schema import Route, Subqueries, QueryChunks


class OpenLLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = AsyncOpenAI(api_key=api_key)


    @override
    async def get_subqueries(self, query: str) -> list[str]:
        response = await self.client.responses.parse(
            model=self.model_name,
            instructions=DECOMPOSITION_INSTRUCTIONS,
            input=query,
            temperature=0.0,
            text_format=Subqueries
        )

        # LLM outputs Q4 sometimes, which is present in FORM-10K not FORM-10Q
        fixed_subqueries = [
            subquery.replace("Q4", "Q3")
            for subquery in response.output_parsed.subqueries
        ]

        return fixed_subqueries


    @override
    async def get_route(self, query: str) -> Route:
        response = await self.client.responses.parse(
            model=self.model_name,
            instructions=ROUTING_INSTRUCTIONS,
            input=query,
            temperature=0.0,
            text_format=Route
        )

        return response.output_parsed


    @override
    async def get_answer_single(self, query_chunks: QueryChunks) -> str:
        if not query_chunks.chunks:
            return "I couldn't retrieve relevant information to answer your question."

        source = query_chunks.chunks[0].filename
        payload = f"# {source}\n"

        for chunk in query_chunks.chunks:
            payload += chunk.content + '\n\n'

        payload += '# Question\n' + query_chunks.query + '\n\n# Answer'

        response = await self.client.responses.create(
            model=self.model_name,
            instructions=SINGLE_ANSWER_GENERATION_INSTRUCTIONS,
            input=payload,
            max_output_tokens=256,
            temperature=0.0
        )

        return response.output_text + ' Source: ' + source


    @override
    async def get_answer_multiple(self, initial_query: str, query_chunks: list[QueryChunks]) -> str:
        if not query_chunks:
            return "I couldn't retrieve relevant information to answer your question."

        payload = "# Supporting context\n"

        for query_chunk in query_chunks:
            payload += f"## {query_chunk.query}\n"
            payload += await self.get_answer_single(query_chunk) + '\n\n'

        payload += '# Question\n' + initial_query + '\n\n# Answer'

        response = await self.client.responses.create(
            model=self.model_name,
            instructions=MULTI_ANSWER_GENERATION_INSTRUCTIONS,
            input=payload,
            max_output_tokens=256,
            temperature=0.0
        )

        print(payload)

        return response.output_text
