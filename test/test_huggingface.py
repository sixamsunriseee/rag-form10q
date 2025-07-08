#Run on colab with GPU
import pandas as pd
import time
from transformers import pipeline
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generation_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    device=device,
    torch_dtype=torch.float16
)

ideal_df = pd.read_csv('/content/ideal_answers.csv')
system_df = pd.read_csv('/content/answers.csv')

ideal_answers = ideal_df['Answer'].tolist()
system_answers = system_df['Answer'].tolist()

results = []
batch_size = 10

def parse_verdict(output_text):
    if "True" in output_text:
        return True
    elif "False" in output_text:
        return False
    else:
        return False 

for i in range(0, len(ideal_df), batch_size):
    batch_ideal = ideal_answers[i:i+batch_size]
    batch_system = system_answers[i:i+batch_size]
    pairs = [
        {"ideal": ideal, "system": system}
        for ideal, system in zip(batch_ideal, batch_system)
    ]
    user_content = (
        "Evaluate the correctness of the system's answer compared to the reference. "
        "Return verdicts: a list of True/False for each pair.\nPairs:\n" + str(pairs)
    )

    response = generation_pipeline(user_content, max_new_tokens=100)[0]['generated_text']
    import re
    print(response)
    verdicts = re.findall(r'\bTrue\b|\bFalse\b', response)
    verdicts = [v == "True" for v in verdicts]
    if len(verdicts) != len(pairs):
        verdicts = [False] * len(pairs)
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