import glob
import re
from pprint import pprint
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


import numpy as np
import pandas as pd
import seaborn as sns

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


fold = KFold(n_splits=5, shuffle=True, random_state=0)
for i, (train_index, test_index) in enumerate(fold.split(data, data.has_citation)):
    test_df = data.iloc[test_index]
    train_df = data.iloc[train_index]

    # for now just grab the first fold
    break


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
    hidden_units=[1000, 500, 250],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

out = []
for _ in range(200):
    estimator.train(input_fn=train_input_fn, steps=500)
    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)
    out.append(test_eval_result)
    print(test_eval_result)
pprint(out)
