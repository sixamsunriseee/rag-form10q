import asyncio

import numpy as np
import pandas as pd
from pydantic import BaseModel
from openai import AsyncOpenAI

from config import OPENAI_KEY


class Verdict(BaseModel):
    conclusion: bool
    figures: bool


async def give_verdict(client: AsyncOpenAI, reference: str, system: str) -> Verdict:
    response = await client.responses.parse(
        model="gpt-4o-mini",
        instructions="""
            Check if figures (e.g. sale numbers) and overall conclusion of system answer correspond to reference.
            
            Rules:
            - Allow rounding in figures.
            - Ignore figures that are present in system answer, but are not present in reference.
            
            Return:
            - Conclusion as True if conclusions correspond.
            - Figures as True if figures correspond.
        """,
        input=str({"reference": reference, "system": system}),
        temperature=0,
        text_format=Verdict,
    )

    return response.output_parsed


async def main():
    client = AsyncOpenAI(api_key=OPENAI_KEY)

    df = pd.read_csv('../data/system_answers/hybrid-top5.csv')

    ideal_answers = df['Answer'].tolist()
    system_answers = df['System Answer'].tolist()

    tasks = [
        give_verdict(client, ideal, system)
        for ideal, system in zip(ideal_answers, system_answers)
    ]

    verdicts = await asyncio.gather(*tasks)

    df['Meaning'] = [verdict.conclusion for verdict in verdicts]
    df['Figures'] = [verdict.figures for verdict in verdicts]

    df.to_csv('../data/system_answers/verdicts.csv', index=False)

    print(np.where(df['Meaning'], 1, 0).mean())
    print(np.where(df['Figures'], 1, 0).mean())


if __name__ == '__main__':
    asyncio.run(main())
