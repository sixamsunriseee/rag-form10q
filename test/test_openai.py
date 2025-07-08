import pandas as pd
import openai
import time
from pydantic import BaseModel

class BatchVerdict(BaseModel):
    verdicts: list[bool]

client = openai.OpenAI(api_key="YOUR_API_KEY")

ideal_df = pd.read_csv('Intership Vention/RAG task/ideal_answers.csv')
system_df = pd.read_csv('Intership Vention/RAG task/answers.csv')

ideal_answers = ideal_df['Answer'].tolist()
system_answers = system_df['Answer'].tolist()

results = []

batch_size = 10

for i in range(0, len(ideal_df), batch_size):
    batch_ideal = ideal_answers[i:i+batch_size]
    batch_system = system_answers[i:i+batch_size]
    pairs = [
        {"ideal": ideal, "system": system}
        for ideal, system in zip(batch_ideal, batch_system)
    ]
    user_content = "Evaluate the correctness of the system's answer compared to the reference. Return verdicts: a list of True/False for each pair.\nPairs:\n" + str(pairs)

    response = client.responses.parse(
        model="gpt-4o-mini-2024-07-18",
        input=[{"role": "user", "content": user_content}],
        temperature=0,
        text_format=BatchVerdict
    )
    verdicts = response.output_parsed.verdicts
    results.extend(verdicts)
    print(f"Processed batch {i//batch_size + 1}/{len(ideal_answers)//batch_size + 1}")
    time.sleep(1)

system_df['results'] = results
system_df.to_csv('results.csv', index=False)

true_count = system_df['results'].value_counts().get(True, 0)
false_count = system_df['results'].value_counts().get(False, 0)
print(f"True count: {true_count}")
print(f"False count: {false_count}")
accuracy = true_count / (true_count + false_count) if (true_count + false_count) > 0 else 0
print(f"Accuracy: {accuracy:.2f}")