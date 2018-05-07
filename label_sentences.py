import pandas as pd


def main():
    """Driver"""
    with open('sentences.csv', 'r') as f:
        data = f.read()
    df = pd.read_csv('sentences.csv')
    for i, row in df.iterrows():
        print(row[0])
        x = input()
        if x:
            df.iloc[i, 1] = 1
    df.to_csv('labeled_sentences.csv', index=False)

if __name__ == '__main__':
    main()  