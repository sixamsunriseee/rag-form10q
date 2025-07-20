from typing import Iterable

import transformers
import torch


class Qwen:
    def __init__(self, *instructions: str):
        self.instructions = [{'role': 'system', 'content': i} for i in instructions]
        self.pipeline = transformers.pipeline(
        "text-generation",
            model="Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=torch.float16
        )

    def answer_from_chunks(self, chunks: Iterable[str], query: str):
        chunks = [{'role': 'system', 'content': chunk} for chunk in chunks]
        query = [{'role': 'user', 'content': query}]

        response = self.pipeline(
            self.instructions + chunks + query,
            max_new_tokens=128
        )


        return response[0]['generated_text'][-1]['content']