import pandas as pd
import numpy as np

def label_sentences(args, filename='sample_paper.csv', label_all=False):
    """Driver"""
    df = pd.read_csv('{}/{}'.format(args.load_from, filename), encoding='utf-8', quotechar='`')
    for i, row in df.iterrows():

        print(row[0])
        # If label_all is True, never skip.
        if not label_all and not np.isnan(row[1]):
            continue

        savequit = False
        while True:
            x = input()
            if x:
                if x == 'q':
                    savequit = True
                    break
                try:
                    df.iloc[i, 1] = int(x)
                    break
                except ValueError:
                    print('Please enter a number or the character q to quit')
            else:
                df.iloc[i, 1] = 0
                break
        if savequit:
            break
    df.to_csv('labeled_sentences/{}'.format(filename), index=False, quotechar='`')


def parse():
    """parse CLI args"""
    import argparse
    parser = argparse.ArgumentParser(description='Allows a human to label sentences via CLI.')
    parser.add_argument('--load_from', default='sentences', help='The directory to load sentences.csv from')
    args = parser.parse_args()
    label_sentences(args)

if __name__ == '__main__':
    parse()