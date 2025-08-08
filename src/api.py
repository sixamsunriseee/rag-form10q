import json

from fastapi import FastAPI

from src.embedding.fastembed_ import MiniLmEmbedding
from src.inference import run_inference
from src.llm.openai_ import OpenLLM
from src.vec_database.dense import DenseDatabase
from config import OPENAI_KEY, DENSE_CONN_STRING, BEST_COLLECTION_NAME

app = FastAPI()
llm = OpenLLM(model_name='gpt-4.1-mini', api_key=OPENAI_KEY)

with open('instructions.json') as file:
    instructions = str(json.load(file))

database = DenseDatabase(conn_string=DENSE_CONN_STRING, dense=MiniLmEmbedding())


@app.get("/generate/")
async def get(query: str):
    generated, chunks = await run_inference(
        llm=llm,
        instructions=instructions,
        database=database,
        collection_name=BEST_COLLECTION_NAME,
        query=query,
        retrieve_limit=20
    )

    return {"query": query, "generated": generated}
