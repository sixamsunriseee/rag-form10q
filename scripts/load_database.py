import json
import os
import asyncio

from src.embedding.fastembed_ import FastEmbedding
from src.vec_database.hybrid import HybridDatabase
from src.vec_database.dense import DenseDatabase
from src.vec_database.base import BaseDatabase
from src.schema import Chunk


async def main(database: BaseDatabase, chunk_file: str, force: bool = False):
    collection_name = os.path.basename(chunk_file).removesuffix('.json')

    if force and await database.client.collection_exists(collection_name):
        await database.client.delete_collection(collection_name)

    await database.create_collection(collection_name)

    with open(chunk_file) as file:
        chunks = (Chunk(**chunk) for chunk in json.load(file))

    await database.upsert_chunks(collection_name, chunks)


if __name__ == '__main__':
    # asyncio.run(main(HybridDatabase(FastEmbedding()), '../data/chunks/markdown-4096-512.json', True))
    # asyncio.run(main(HybridDatabase(FastEmbedding()), '../data/chunks/text-4096-512.json', True))
    asyncio.run(main(DenseDatabase(FastEmbedding()), '../data/chunks/markdown-4096-512.json', True))
    asyncio.run(main(DenseDatabase(FastEmbedding()), '../data/chunks/text-4096-512.json', True))
