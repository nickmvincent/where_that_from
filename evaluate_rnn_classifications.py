"""
original source: https://github.com/crazydonkey200/tensorflow-char-rnn/blob/master/sample.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import argparse
import codecs
import json
import os

import numpy as np
import pandas as pd
from sklearn import metrics
from char_rnn_model import *
from train import load_vocab

def main():
    parser = argparse.ArgumentParser()
    
    # Parameters for using saved best models.
    parser.add_argument('--init_dir', type=str, default='',
                        help='continue from the outputs in the given directory')

    # Parameters for picking which model to use. 
    parser.add_argument('--model_path', type=str, default='',
                        help='path to the model file like output/best_model/model-40.')

    # Parameters for sampling.
    parser.add_argument('--temperature', type=float,
                        default=1.0,
                        help=('Temperature for sampling from softmax: '
                              'higher temperature, more random; '
                              'lower temperature, more greedy.'))

    parser.add_argument('--max_prob', dest='max_prob', action='store_true',
                        help='always pick the most probable next character in sampling')

    parser.set_defaults(max_prob=False)

    parser.add_argument('--seed', type=int,
                        default=-1,
                        help=('seed for sampling to replicate results, '
                              'an integer between 0 and 4294967295.'))

    # Parameters for debugging.
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='show debug information')
    parser.set_defaults(debug=False)
    
    args = parser.parse_args()

    # Prepare parameters.
    with open(os.path.join(args.init_dir, 'result.json'), 'r') as f:
        result = json.load(f)
    params = result['params']

    if args.model_path:    
        best_model = args.model_path
    else:
        best_model = result['best_model']

    best_valid_ppl = result['best_valid_ppl']
    if 'encoding' in result:
        args.encoding = result['encoding']
    else:
        args.encoding = 'utf-8'
    args.vocab_file = os.path.join(args.init_dir, 'vocab.json')
    vocab_index_dict, index_vocab_dict, vocab_size = load_vocab(args.vocab_file, args.encoding)


    # Create graphs
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('evaluation'):
            test_model = CharRNN(is_training=False, use_batch=False, **params)
            saver = tf.train.Saver(name='checkpoint_saver')

    if args.seed >= 0:
        np.random.seed(args.seed)
    # Sampling a sequence 

    data = pd.read_csv('data/test_0.csv', encoding='utf8')

    y = []
    y_hat = []
    n = len(data.index)
    
    with tf.Session(graph=graph) as session:
        saver.restore(session, best_model)
        
        for i, row in data.iterrows():
            if i % 10 == 0:
                print('{} of {} examples tested'.format(i, n))
            text = row['processed_text'] + '</s>'
            has_citation = int(row['has_citation'])
            y.append(has_citation)
            sample = test_model.sample_seq(session, 1, text,
                                            vocab_index_dict, index_vocab_dict,
                                            temperature=args.temperature,
                                            max_prob=args.max_prob)
            try:
                predicted_label = int(sample[-1])
            except ValueError:
                predicted_label = 0
            y_hat.append(predicted_label)

    try:
        roc_auc = metrics.roc_auc_score(y, y_hat)
    except ValueError:
        roc_auc = 'undefined'
    f1_macro = metrics.f1_score(y, y_hat, average='macro')
    acc = metrics.accuracy_score(y, y_hat)

    print('roc_auc: {}\nf1_macro:{}\nacc:{}'.format(
        roc_auc, f1_macro, acc
    ))

    df = pd.DataFrame()
    df['y'] = y
    df['y_hat'] = y_hat
    df.to_csv(args.init_dir + '/predictions.csv')
    with open('y_hat.txt', 'w') as f:
        f.write('\n'.join(
            [str(x) for x in y_hat]
        ))

if __name__ == '__main__':
    main()