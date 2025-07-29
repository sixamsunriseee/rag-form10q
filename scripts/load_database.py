import json
import os

from src.vec_database import QdrantDatabase
from src.schema.chunk import Chunk


def main(chunk_file: str, force: bool = False):
    collection_name = os.path.basename(chunk_file).removesuffix('.json')

    database = QdrantDatabase()

    if force and database.client.collection_exists(collection_name):
        database.client.delete_collection(collection_name)

    database.create_collection(collection_name)

    with open(chunk_file) as file:
        chunks = (Chunk(**chunk) for chunk in json.load(file))

    database.upsert_chunks(collection_name, chunks)


if __name__ == '__main__':
    # main('../data/chunks/markdown-4096-512.json', True)
    main('../data/chunks/text-4096-512.json', True)
