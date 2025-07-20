import os

from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from llm import Qwen
from src.document_processors import PlumberProcessor
from src.database import QdrantDatabase


doc_processor = PlumberProcessor()

dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding("Qdrant/bm25")
late_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

collection_name = 'form10q'
database = QdrantDatabase(
    '../data/qdrant/',
    dense_model=dense_model,
    sparse_model=sparse_model,
    late_model=late_model
)

llm = Qwen(
    'You are a bot, whose objective is to answer user query with the provided FORM-10Q data from the system'
)


def clean_load_db():
    database.create_collection(collection_name)
    docs_path = '../data/docs/'

    for file in os.listdir(docs_path):
        if not file.endswith('.pdf'):
            continue

        file = docs_path + file
        chunks = doc_processor.to_chunks(file)
        database.upsert_chunks(collection_name, chunks)
        print(file)


def main():
    clean_load_db()

    query = "In the first quarter of 2023, how much did Apple spend on research and development, and what was the focus of this expenditure?"
    points = database.query(collection_name, query).points

    response = llm.answer_from_chunks(
        map(lambda x: str(x.payload), points),
        query
    )

    print(response)


if __name__ == '__main__':
    main()