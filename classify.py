"""
try to classify reddit posts.
"""
import os
import pandas as pd
import glob
from sklearn_pandas import DataFrameMapper, cross_val_score
import numpy as np
from collections import defaultdict
from pprint import pprint
import time

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def print_topk(k, feature_names, clf):
    """Prints features with the highest coefficient values, per class"""
    topk = np.argsort(clf.coef_[0])[-k:]
    print(
        "{}".format(
            " ".join(feature_names[j] for j in topk[::-1])
        )
    )

def main():
    if False:
        path = 'labeled_sentences/vincent_etal_chi_2018.csv'
        data = pd.read_csv(path, encoding='utf-8')
    else:
        sentences_filepaths = glob.glob("labeled_sentences/*.csv")
        data = None
        for path in sentences_filepaths:
            if data is None:
                data = pd.read_csv(path, encoding='utf-8')
            else:
                data = pd.concat([data, pd.read_csv(path, encoding='utf-8')])
        data.to_csv('all_labeled_sentences.csv')
    


    # "Feature Engineering"
    data['length'] = data.apply(lambda row: len(row['processed_text']), axis=1)
    data['has_digits'] = data.apply(lambda row: any(char.isdigit() for char in row['processed_text']), axis=1)
    # with open("glove.6B.50d.txt", "rb") as lines:
    #     w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
    #         for line in lines}

    print(data.head())
    print(len(data.index))
    #input()

    mapper = DataFrameMapper([
        ('processed_text', 
            CountVectorizer(
                #stop_words='english', 
                lowercase=True,
                #ngram_range=(5,5)
                # max_features=10000
                )
        ),
        (['length'], StandardScaler()),
        ('has_digits', LabelBinarizer())
    ])
    X = mapper.fit_transform(data.copy())

    algo_to_score = defaultdict(dict)
    for clf, name in [
        (DummyClassifier(strategy='most_frequent'), 'dummy',),
        (LogisticRegression(verbose=0), 'logistic'),
        (svm.LinearSVC(verbose=0), 'svm',),
        (ExtraTreesClassifier(), 'extra_trees'),
        (DecisionTreeClassifier(), 'tree'),
        #(KNeighborsClassifier(), '5nn'), # very slow!
        (GaussianNB(), 'GaussianNB', ) # 'test_accuracy': 0.6799676150821188
    ]:
        start = time.time()
        # if name == 'logistic':
        #     clf.fit(X, data.has_citation)
        #     print_topk(10, mapper.transformed_names_, clf)
        for i, fold in enumerate([
            # StratifiedKFold(5, True, 0),
            KFold(n_splits=5, shuffle=True, random_state=0)
        ]):
            scores = cross_validate(
                clf, X, y=data.has_citation, cv=fold,
                scoring=['accuracy', 'roc_auc', 'f1_macro', ])
            ret = {}
            for key, val in scores.items():
                if 'test_' in key:
                    ret[key] = np.mean(val)
            algo_to_score[name + str(i)] = ret
        print(name, time.time() - start)
    pprint(algo_to_score)
    result_df = pd.DataFrame(algo_to_score)
    print(result_df)

if __name__ == '__main__':
    main()