import asyncio
import time

import pandas as pd

from src.embedding.fastembed_ import FastEmbedding
from src.inference import run_inference
from src.vec_database.dense import DenseDatabase


async def main():
    df = pd.read_csv('../data/ideal_answers.csv')
    questions = df['Question'].tolist()

    database = DenseDatabase(FastEmbedding())

    tasks = []
    answers = []

    for i, question in enumerate(questions):
        tasks.append(run_inference(database, 'text-4096-512', question))

        if (i + 1) % 10 == 0 or i == len(questions) - 1:
            answers.extend(await asyncio.gather(*tasks))
            tasks.clear()
            time.sleep(60)

    df['Answer'] = answers

    df.to_csv('../data/text-4096-512-minilm-gpt-4.1-nano (2).csv', index=False)


if __name__ == '__main__':
    asyncio.run(main())