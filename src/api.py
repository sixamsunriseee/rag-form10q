import os

import uvicorn
import dotenv
from fastapi import FastAPI
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding

from src.embedding.fastembed_ import MiniLmEmbedding
from src.inference import run_inference
from src.llm.openai_ import OpenLLM
from src.vec_database.hybrid import HybridDatabase


dotenv.load_dotenv()

app = FastAPI()

database = HybridDatabase(
    MiniLmEmbedding(),
    SparseTextEmbedding(os.getenv("SPARSE_MODEL")),
    LateInteractionTextEmbedding(os.getenv("LATE_INTERACTION_MODEL"))
)

llm = OpenLLM()


@app.get("/generate/")
async def generate(query: str):
    generated = await run_inference(
        database=database,
        collection_name=os.getenv("COLLECTION_NAME"),
        retrieve_limit=5,
        llm=llm,
        query=query
    )

    return {"query": query, "generated": generated}


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
