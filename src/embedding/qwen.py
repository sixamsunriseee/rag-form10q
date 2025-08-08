from typing import override

import numpy as np
from sentence_transformers import SentenceTransformer

from src.embedding.base import BaseEmbedding


class QwenEmbeddingSmall(BaseEmbedding):
    def __init__(self):
        super().__init__(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            embedding_size=1024
        )

        self.model = SentenceTransformer(self.model_name)


    @override
    async def embed(self, content: str) -> list[float] | np.ndarray:
        return self.model.encode(content).tolist()
