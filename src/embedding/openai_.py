from typing import override

import numpy as np
from openai import AsyncOpenAI

from src.embedding.base import BaseEmbedding


class OpenEmbeddingSmall(BaseEmbedding):
    def __init__(self):
        super().__init__(
            model_name='text-embedding-3-small',
            embedding_size=1536
        )

        self.client = AsyncOpenAI()


    @override
    async def embed(self, query: str) -> list[float] | np.ndarray:
        response = await self.client.embeddings.create(
            input=query,
            model=self.model_name,
        )

        return response.data[0].embedding
