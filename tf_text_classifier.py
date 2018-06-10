import glob
import re
from pprint import pprint
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import time

import numpy as np
import pandas as pd
import seaborn as sns

best_roc_aucs = {
    'accuracy': [],
    'accuracy_baseline': [],
    'auc': [],
    'steps': [],
    'time': [],
}
best_accuracies = {
    'accuracy': [],
    'accuracy_baseline': [],
    'auc': [],
    'steps': [],
    'time': [],
}

data_dir = 'psa_research'
sentences_filepaths = glob.glob(
    "labeled_sentences/{}/*.csv".format(data_dir)
)
data = None
count = 100


for i, path in enumerate(sentences_filepaths):
    if i > count:
        break
    if data is None:
        data = pd.read_csv(path, encoding='utf-8')
    else:
        data = pd.concat([data, pd.read_csv(path, encoding='utf-8')])

tf.logging.set_verbosity('ERROR')


fold = KFold(n_splits=5, shuffle=True, random_state=0)
for i, (train_index, test_index) in enumerate(fold.split(data, data.has_citation)):
    print('Fold Number:', i)
    best_roc_auc = {}
    best_accuracy = {}
    test_df = data.iloc[test_index]
    train_df = data.iloc[train_index]
    # Training input on the whole training set with no limit on training epochs.
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["has_citation"], num_epochs=None, shuffle=True)

    # Prediction on the whole training set.
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["has_citation"], shuffle=False)
    # Prediction on the test set.
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, test_df["has_citation"], shuffle=False)

    embedded_text_feature_column = hub.text_embedding_column(
        key="processed_text", 
        #module_spec="https://tfhub.dev/google/nnlm-en-dim128/1"
        module_spec="https://tfhub.dev/google/Wiki-words-250/1"
    )

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[250, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=2,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
        dropout=0.1,
        model_dir='tf_model_{}'.format(i)
    )
    start = time.time()

    out = []
    for epoch in range(25):
        print('epoch', epoch)
        estimator.train(input_fn=train_input_fn, steps=1000)
        train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
        test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)
        out.append(test_eval_result)
        print(test_eval_result)
        tic = time.time()
        if test_eval_result['auc'] > best_roc_auc.get('auc', 0):
            best_roc_auc = {
                'auc': test_eval_result['auc'],
                'accuracy': test_eval_result['accuracy'],
                'accuracy_baseline': test_eval_result['accuracy_baseline'],
                'steps': test_eval_result['global_step'],
                'time': tic - start,
            }
            
        if test_eval_result['accuracy'] > best_accuracy.get('accuracy', 0):
            best_accuracy = {
                'auc': test_eval_result['auc'],
                'accuracy': test_eval_result['accuracy'],
                'accuracy_baseline': test_eval_result['accuracy_baseline'],
                'steps': test_eval_result['global_step'],
                'time': tic - start,
            }
    for d, d_o_l in (
        (best_roc_auc, best_roc_aucs),
        (best_accuracy, best_accuracies),
    ):
        for key in ['accuracy', 'accuracy_baseline', 'auc', 'steps', 'time']:
            d_o_l[key].append(d[key])
    pprint(out)

print(best_roc_aucs)
print(best_accuracies)

best_roc_auc_row = {
    'accuracy': np.mean(best_roc_aucs['accuracy']),
    'accuracy_std': np.std(best_roc_aucs['accuracy']),
    'algo_name': 'DNN', 
    'max_features': None,
    'feature_selector': None,
    'weights': None,
    'name': 'DNN_{}steps'.format(int(np.mean(best_roc_aucs['steps']))), 
    'roc_auc': np.mean(best_roc_aucs['auc']),
    'roc_auc_std': np.std(best_roc_aucs['auc']),
    'time': np.mean(best_roc_aucs['time']),
}
best_accuracy_row = {
    'accuracy': np.mean(best_accuracies['accuracy']),
    'accuracy_std': np.std(best_accuracies['accuracy']),
    'algo_name': 'DNN', 
    'max_features': None,
    'feature_selector': None,
    'weights': None,
    'name': 'DNN_{}steps'.format(int(np.mean(best_accuracies['steps']))), 
    'roc_auc': np.mean(best_accuracies['auc']),
    'roc_auc_std': np.std(best_accuracies['auc']),
    'time': np.mean(best_accuracies['time']),
}
pd.DataFrame([best_roc_auc_row]).to_csv(
    'results/{}/dnn_best_roc_auc.csv'.format(data_dir)
)
pd.DataFrame([best_accuracy_row]).to_csv(
    'results/{}/dnn_best_accuracy.csv'.format(data_dir)
)