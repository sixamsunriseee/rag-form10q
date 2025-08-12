import uvicorn
from fastapi import FastAPI
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding

from src.embedding.fastembed_ import MiniLmEmbedding
from src.inference import run_inference
from src.llm.openai_ import OpenLLM
from src.vec_database.hybrid import HybridDatabase
from config import OPENAI_KEY, HYBRID_CONN_STRING, BEST_COLLECTION_NAME

app = FastAPI()

database = HybridDatabase(
    HYBRID_CONN_STRING,
    MiniLmEmbedding(),
    SparseTextEmbedding("Qdrant/bm25"),
    LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
)

llm = OpenLLM(model_name='gpt-4.1-nano', api_key=OPENAI_KEY)

@app.get("/generate/")
async def generate(query: str):
    generated = await run_inference(
        database=database,
        collection_name=BEST_COLLECTION_NAME,
        retrieve_limit=5,
        llm=llm,
        query=query
    )

    return {"query": query, "generated": generated}


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
