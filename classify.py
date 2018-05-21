"""
try to classify reddit posts.
"""
import os
import pandas as pd
import glob
from sklearn_pandas import DataFrameMapper, cross_val_score
import sklearn
import numpy as np
from collections import defaultdict
from pprint import pprint

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid


def print_topk(k, feature_names, clf):
    """Prints features with the highest coefficient values, per class"""
    topk = np.argsort(clf.coef_[0])[-k:]
    print(
        "{}".format(
            " ".join(feature_names[j] for j in topk[::-1])
        )
    )

def main():
    filename = 'vincent_etal_chi_2018.csv'
    path = 'labeled_sentences/' + filename
    data = pd.read_csv(path, encoding='utf-8')

    mapper = DataFrameMapper([
        ('processed_text', CountVectorizer(
            #stop_words='english', 
            lowercase=True,
            # max_features=10000
            )
        ),
    ])
    X = mapper.fit_transform(data.copy())

    algo_to_score = defaultdict(dict)
    for clf, name in [
        (svm.LinearSVC(verbose=0), 'svm',),
        (SGDClassifier(max_iter=10, tol=None, verbose=0), 'sgd',),
        (LogisticRegression(verbose=0), 'logistic'),
    ]:
        if name == 'logistic':
            clf.fit(X, data.has_citation)
            print_topk(20, mapper.transformed_names_, clf)
        for i, folds in enumerate([
            # StratifiedKFold(5, True, 0),
            KFold(n_splits=5, shuffle=True, random_state=0)
        ]):
            scores = cross_validate(
                clf, X, y=data.has_citation, cv=folds,
                scoring=['f1', 'precision', 'recall',])
            ret = {}
            for key, val in scores.items():
                if 'test_' in key:
                    ret[key] = np.mean(val)
            algo_to_score[name + str(i)]['cross_validation'] = ret
    pprint(algo_to_score)

if __name__ == '__main__':
    main()