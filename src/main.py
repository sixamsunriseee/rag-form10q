import os

from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from llm import Qwen
from src.document_processors import PlumberProcessor
from src.database import QdrantDatabase


doc_processor = PlumberProcessor()

dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding("Qdrant/bm25")
late_model = LateInteractionTextEmbedding("jinaai/jina-colbert-v2")
reranker = TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual')

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
    # clean_load_db()

    query = "In the first quarter of 2023, how much did Apple spend on research and development, and what was the focus of this expenditure?"
    points = database.query(collection_name, query).points
    points.sort(key=lambda x: x.score, reverse=True)
    hits = [str(x.payload) for x in points]

    print('Initial Retrieval:')
    for i, hit in enumerate(hits):
        print(i, hit)

    points = reranker.rerank(query, hits)

    ranking = [
        (i, score) for i, score in enumerate(points)
    ]

    ranking.sort(
        key=lambda x: x[1], reverse=True
    )

    new_hits = []
    print('Reranked')
    for i, score in ranking[:20]:
        print(i, hits[i])
        new_hits.append(hits[i])

    response = llm.answer_from_chunks(
        new_hits[:10],
        query
    )

    print(response)


if __name__ == '__main__':
    main()