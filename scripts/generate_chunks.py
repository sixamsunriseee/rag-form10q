import os
import json

from src.chunker.text import TextChunker
from src.schema import Chunk
from config import TEXTS_PATH, CHUNKS_DEFAULT_PATH


def main(size: int, overlap: int, parsed_directory: str, output: str):
    chunker = TextChunker(chunk_size=size, chunk_overlap=overlap)
    chunks = list()

    for filename in os.listdir(parsed_directory):
        with open(os.path.join(parsed_directory, filename)) as parsed_file:
            contents = parsed_file.read()

        file_chunks = (Chunk(filename=filename, text=chunk) for chunk in chunker.split_to_chunks(contents))

        chunks.extend(chunk.model_dump() for chunk in file_chunks)

    with open(output, 'w') as file:
        json.dump(chunks, file, indent=4)


if __name__ == '__main__':
    # main(4096, 512, MARKDOWNS_PATH, CHUNKS_DEFAULT_PATH + 'markdown-4096-512.json')
    # main(4096, 512, TEXTS_PATH, CHUNKS_DEFAULT_PATH + 'text-4096-512.json')
    main(2048, 256, TEXTS_PATH, CHUNKS_DEFAULT_PATH + 'text-2048-256.json')
    main(1024, 128, TEXTS_PATH, CHUNKS_DEFAULT_PATH + 'text-1024-128.json')
