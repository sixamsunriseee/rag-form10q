from abc import ABC, abstractmethod
from typing import Iterable

import transformers
import torch
from openai_ import OpenAI, AsyncOpenAI

from src.schema import Chunk, ChunkWithContext


class GptLanguageModel:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def answer(self, instructions: str, query: str) -> str:
        response = self.client.responses.create(
            model='gpt-4.1-nano',
            instructions=instructions,
            input=query,
            max_output_tokens=256
        )

        return response.output_text


class QwenLanguageModel:
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
            self.instructions + chunks + query
        )

        return response[0]['generated_text'][-1]['content']