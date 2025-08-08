from typing import override

import numpy as np
from openai import AsyncOpenAI

from src.embedding.base import BaseEmbedding


class OpenEmbeddingSmall(BaseEmbedding):
    def __init__(self, api_key: str):
        super().__init__(
            model_name='text-embedding-3-small',
            embedding_size=1536
        )

        self.client = AsyncOpenAI(api_key=api_key)


    @override
    async def embed(self, content: str) -> list[float] | np.ndarray:
        response = await self.client.embeddings.create(
            input=content,
            model=self.model_name,
        )

        return response.data[0].embedding
