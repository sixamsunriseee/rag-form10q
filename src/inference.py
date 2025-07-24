import os

from src.doc_processors import BaseProcessor
from src.language_model import QwenLanguageModel
from src.schema.pipeline_config import PipelineConfig
from src.vec_database import QdrantDatabase


language_model = QwenLanguageModel(
    'You are a bot, whose objective is to answer user query with the provided FORM-10Q data from the system'
)


def clean_load_db(database: QdrantDatabase, processor: BaseProcessor, collection_name: str):
    if database.client.collection_exists(collection_name):
        database.client.delete_collection(collection_name)

    database.create_collection(collection_name)
    docs_path = '../data/docs/'

    for file in os.listdir(docs_path):
        if not file.endswith('.pdf'):
            continue

        prefetch = processor.parse_to_string(docs_path + file)
        chunks = processor.split_to_chunks(docs_path + file, prefetch)
        database.upsert_chunks(collection_name, chunks)

        print(file)


def run_inference(config: PipelineConfig):
    collection_name = 'form10q'
    database = QdrantDatabase(url='../data/qdrant/', embeddings=config.embeddings)

    # clean_load_db(database, config.processor, collection_name)

    query = "How has Apple's total net sales changed over time?"
    points = list(database.query(collection_name, query))


    response = language_model.answer_from_chunks(
        points,
        query
    )

    print(response)


if __name__ == '__main__':
    from src.pipelines import pipeline_fast
    run_inference(pipeline_fast)
