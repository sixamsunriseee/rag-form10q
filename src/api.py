from fastapi import FastAPI
from src.inference import run_inference

app = FastAPI()

@app.get("/predict/markdown/")
async def read_root(query: str):
    return {"answer": run_inference('markdown-4096-512', query)}

@app.get("/predict/text/")
async def read_root(query: str):
    return {"answer": run_inference('text-4096-512', query)}
