from abc import ABC, abstractmethod
from typing import Iterable

import transformers
import torch

from src.schema.chunk import Chunk, ChunkWithContext


class QwenLanguageModel:
    def __init__(self, *instructions: str):
        self.instructions = [{'role': 'system', 'content': i} for i in instructions]
        self.pipeline = transformers.pipeline(
        "text-generation",
            model="Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=torch.float16
        )

    def generate_context(self, filename: str, doc_text: str, chunk: Chunk) -> ChunkWithContext:
        prompt = f'''
            <document>
            {filename}
            {doc_text}
            </document> 
            Here is the chunk we want to situate within the whole document 
            <chunk> 
            {chunk.text} 
            </chunk> 
            Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 
        '''

        query = [{'role': 'system', 'content': prompt}]

        response = self.pipeline(
            query,
            max_new_tokens=128
        )

        return ChunkWithContext(filename=filename, text=chunk.text, context=response[0]['generated_text'][-1]['content'])

    def answer_from_chunks(self, chunks: Iterable[str], query: str):
        chunks = [{'role': 'system', 'content': chunk} for chunk in chunks]
        query = [{'role': 'user', 'content': query}]

        response = self.pipeline(
            self.instructions + chunks + query,
            max_new_tokens=128
        )

        return response[0]['generated_text'][-1]['content']