import asyncio

import pandas as pd
from pydantic import BaseModel
from openai import AsyncOpenAI

from config import OPENAI_KEY


class Verdict(BaseModel):
    verdict: bool


async def give_verdict(client: AsyncOpenAI, reference: str, system: str) -> bool:
    response = await client.responses.parse(
        model="gpt-4o-mini",
        instructions="Evaluate the correctness of the system's answer compared to the reference.",
        input=str({"reference": reference, "system": system}),
        temperature=0,
        text_format=Verdict,
    )

    return response.output_parsed.verdict


async def main():
    client = AsyncOpenAI(api_key=OPENAI_KEY)

    ideal_df = pd.read_csv('../data/ideal_answers.csv')
    system_df = pd.read_csv('../data/system_answers/1024-txt-minilm-dense-gpt-4.1-mini.csv')

    ideal_answers = ideal_df['Answer'].tolist()
    system_answers = system_df['Answer'].tolist()

    tasks = [give_verdict(client, ideal, system) for ideal, system in zip(ideal_answers, system_answers)]

    verdicts = list(await asyncio.gather(*tasks))
    system_df['Verdict'] = verdicts
    system_df.to_csv('1024-txt-minilm-dense-gpt-4.1-mini-verdicts.csv', index=False)

    print(sum(system_df['Verdict']) / len(system_df))


if __name__ == '__main__':
    asyncio.run(main())
