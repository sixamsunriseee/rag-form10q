import os
import json
from typing import Iterable

from src.chunker import TextChunker
from src.schema.chunk import Chunk, ChunkWithContext
from config import MARKDOWNS_PATH, TEXTS_PATH, CHUNKS_DEFAULT_PATH, OPENAI_KEY
from src.language_model import GptLanguageModel


model = GptLanguageModel(OPENAI_KEY)


def contextualize(chunks: Iterable[Chunk]) -> Iterable[ChunkWithContext]:
    prompt = '''
        <document> 
        {{WHOLE_DOCUMENT}} 
        </document> 
        Here is the chunk we want to situate within the whole document 
        <chunk> 
        {{CHUNK_CONTENT}} 
        </chunk> 
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.  
    '''



def main(size: int, overlap: int, parsed_directory: str, output: str, context: bool):
    chunker = TextChunker(chunk_size=size, chunk_overlap=overlap)
    chunks = list()

    for filename in os.listdir(parsed_directory):
        with open(os.path.join(parsed_directory, filename)) as parsed_file:
            contents = parsed_file.read()

        file_chunks = (Chunk(filename=filename, text=chunk) for chunk in chunker(contents))

        if context:
            file_chunks = contextualize(file_chunks)

        chunks.extend(chunk.model_dump() for chunk in file_chunks)


    with open(output, 'w') as file:
        json.dump(chunks, file, indent=4)


if __name__ == '__main__':
    # main(4096, 512, MARKDOWNS_PATH, CHUNKS_DEFAULT_PATH + 'markdown-4096-512.json', False)
    main(4096, 512, TEXTS_PATH, CHUNKS_DEFAULT_PATH + 'text-4096-512.json', False)
