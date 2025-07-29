import pandas as pd

from src.inference import run_inference


def main():
    df = pd.read_csv('../data/questions.csv')
    questions = df['Question'].tolist()

    answers = [run_inference(question) for question in questions]
    df['Answer'] = answers

    df.to_csv('../data/answers.csv', index=False)


if __name__ == '__main__':
    main()