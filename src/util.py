import os
import json
import time
import asyncio
from itertools import batched

from src.llm.base import BaseLLM
from src.parser.base import BaseParser
from src.chunker.text import TextChunker
from src.vec_database.base import BaseDatabase
from src.inference import run_inference
from src.schema import Chunk


def parse_documents(parser: BaseParser, docs_path: str, output_directory: str, suffix: str):
    for filename in os.listdir(docs_path):
        if not filename.endswith('.pdf'):
            continue

        contents = parser.parse_to_string(os.path.join(docs_path, filename))
        filename = filename.removesuffix('.pdf') + suffix

        with open(os.path.join(output_directory, filename), 'w') as parsed_file:
            parsed_file.write(contents)


def generate_chunks(size: int, overlap: int, input_directory: str, input_suffix: str, output_file: str):
    chunker = TextChunker(chunk_size=size, chunk_overlap=overlap)
    chunks = list()

    for filename in os.listdir(input_directory):
        with open(os.path.join(input_directory, filename)) as parsed_file:
            contents = parsed_file.read()

        filename = filename.removesuffix(input_suffix)

        year, quarter, company = filename.split()
        filename += '.pdf'

        file_chunks = (
            Chunk(filename=filename, company=company, year=int(year), quarter=quarter, content=chunk)
            for chunk in chunker.split_to_chunks(contents)
        )

        chunks.extend(chunk.model_dump() for chunk in file_chunks)

    with open(output_file, 'w') as file:
        json.dump(chunks, file, indent=4)


async def load_chunks_to_database(
    database: BaseDatabase,
    collection_name: str,
    chunk_file: str,
    batch_size: int = None,
    batch_delay: float = 60.0,
    force: bool = False
):
    if force and await database.client.collection_exists(collection_name):
        await database.client.delete_collection(collection_name)

    await database.create_collection(collection_name)

    with open(chunk_file) as file:
        chunks = (Chunk(**chunk) for chunk in json.load(file))

    if batch_size:
        for batch in batched(chunks, batch_size):
            await database.upsert_chunks(collection_name, batch)
            time.sleep(batch_delay)
    else:
        await database.upsert_chunks(collection_name, chunks)


async def answer_questions(
    llm: BaseLLM,
    instructions: str,
    database: BaseDatabase,
    collection_name: str,
    retrieve_limit: int,
    questions: list[str],
    batch_size: int,
    batch_delay: float = 60.0
):
    answers = list()

    for batch in batched(questions, batch_size):
        tasks = [
            run_inference(llm, instructions, database, collection_name, question, retrieve_limit)
            for question in batch
        ]
        answers.extend(await asyncio.gather(*tasks))
        time.sleep(batch_delay)

    return answers
