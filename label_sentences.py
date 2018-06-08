import pandas as pd
import numpy as np
import re

def label_sentences(filepath='sample_paper.csv', mode='manual',load_from=None,  label_all=False):
    """Driver"""
    if load_from:
        filepath = '{}/{}'.format(load_from, filepath)
    print('filepath', filepath)
    df = pd.read_csv(filepath, encoding='utf-8')
    for i, row in df.iterrows():
        sentence = row[0]
        # If label_all is True, never skip.
        if mode == 'manual':
            print(sentence)        
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
                        df.loc[i, 'label'] = int(x)
                        break
                    except ValueError:
                        print('Please enter a number or the character q to quit')
                else:
                    df.loc[i, 'label'] = 0
                    break
            if savequit:
                break
        else:
            matches = re.findall(r' ?\[\d.*?\]', sentence)
            if matches:
                df.loc[i, 'has_citation'] = 1
            else:
                df.loc[i, 'has_citation'] = 0
            sent_wo_matches = sentence
            for match in matches:
                sent_wo_matches = sent_wo_matches.replace(match, "")
            #sent_wo_matches = sent_wo_matches.replace(' ,', ',')
            sent_wo_matches = sent_wo_matches.replace(
                    '(eg)', ''
                ).replace(
                    '(eg,)', ''
                )
            
            sent_wo_matches = sent_wo_matches.replace(
                    ' .', '.'
                ).replace(
                    ' ,', ','
                )
            
            # hanging_eg = sent_wo_matches.find('(eg')
            # if hanging_eg != -1:
            #     close = sent_wo_matches.find(')', hanging_eg)
            #     sent_wo_matches = sent_wo_matches[:hanging_eg] + sent_wo_matches[close:]
            df.loc[i, 'processed_text'] = sent_wo_matches
    outname = filepath.replace(
        'sentences/', 'labeled_sentences/',    
    ).replace(
        'sentences\\', 'labeled_sentences/'
    )
    df.to_csv(outname, index=False)


def parse():
    """parse CLI args"""
    import argparse
    parser = argparse.ArgumentParser(description='Allows a human to label sentences via CLI.')
    parser.add_argument('--load_from', default='sentences', help='The directory to load sentences.csv from')
    parser.add_argument('--mode', default='manual', help='manual or auto-detect-citations')
    args = parser.parse_args()
    if args.mode == 'manual':
        label_sentences(load_from=args.load_from, mode=args.mode)

if __name__ == '__main__':
    parse()