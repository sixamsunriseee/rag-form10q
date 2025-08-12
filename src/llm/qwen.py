from typing import  override

import torch
import transformers

from src.llm.base import BaseLLM
from src.schema import Route, QueryChunks


class QwenLLM(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16,
        )

    @override
    async def get_subqueries(self, query: str) -> list[str]: ...

    @override
    async def get_route(self, query: str) -> Route: ...

    @override
    async def get_answer_single(self, query_chunks: QueryChunks) -> str: ...

    @override
    async def get_answer_multiple(self, initial_query: str, query_chunks: list[QueryChunks]) -> str: ...


    # @override
    # async def query(self, instructions: str, chunks: list[Chunk], query: str) -> str:
    #     response = self.pipeline(
    #         [
    #             {"role": "system", "content": str({'instructions': instructions})},
    #             {"role": "system", "content": str({'chunks': chunks})},
    #             {"role": "user", "content": str({'query': query})}
    #         ],
    #         do_sample=False,  # equivalent of temperature=0.0
    #         max_new_tokens=256,
    #     )
    #
    #     return response[0]['generated_text'][-1]['content']
