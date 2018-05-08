import pandas as pd
import numpy as np

def main(filename='sample_paper.csv', load_from='sentences', label_all=False):
    """Driver"""
    df = pd.read_csv('{}/{}'.format(load_from, filename), encoding='utf-8', quotechar='`')
    for i, row in df.iterrows():

        # If label_all is True, never skip.
        if not label_all and not np.isnan(row[1]):
            continue
        x = input()

        if x:
            if x == 'q':
                break
            df.iloc[i, 1] = float(x)
    df.to_csv('labeled_sentences/labeled_{}'.format(filename), index=False, quotechar='`')

if __name__ == '__main__':
    main()  