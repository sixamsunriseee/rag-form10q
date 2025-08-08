from typing import  override

import torch
import transformers

from src.llm.base import BaseLLM
from src.schema import QueryRoute, Chunk


class QwenLLM(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16,
        )


    @override
    async def get_query_route(self, query: str) -> QueryRoute:
        pass


    @override
    async def query(self, instructions: str, chunks: list[Chunk], query: str) -> str:
        response = self.pipeline(
            [
                {"role": "system", "content": str({'instructions': instructions})},
                {"role": "system", "content": str({'chunks': chunks})},
                {"role": "user", "content": str({'query': query})}
            ],
            do_sample=False,  # equivalent of temperature=0.0
            max_new_tokens=256,
        )

        return response[0]['generated_text'][-1]['content']
