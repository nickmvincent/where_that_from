"""
try to classify sentences from a paper.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate, KFold
from sklearn import svm, tree

def print_topk(k, feature_names, clf):
    """Prints features with the highest coefficient values, per class"""
    topk = np.argsort(clf.coef_[0])[-k:]
    print(
        "{}".format(
            " ".join(feature_names[j] for j in topk[::-1])
        )
    )

def main():
    """driver"""    
    data = pd.read_csv('labeled_sentences.csv')

    mapper = DataFrameMapper([
        ('text', TfidfVectorizer(
                stop_words='english', lowercase=True,
            )
        ),
    ])

    X = mapper.fit_transform(data.copy())


    algo_to_score = defaultdict(dict)
    for clf, name in [
        #(svm.LinearSVC(verbose=0), 'svm',),
        (LogisticRegression(verbose=0), 'logistic'),
        (tree.DecisionTreeClassifier(), 'tree'),
    ]:
        if name == 'logistic':
            clf.fit(X, data.label)
            print_topk(20, mapper.transformed_names_, clf)
        for i, folds in enumerate([
            KFold(5, True, 0)
        ]):
            scores = cross_validate(
                clf, X, y=data.label, cv=folds,
                scoring=['f1', 'precision', 'recall',])
            ret = {}
            for key, val in scores.items():
                if 'test_' in key:
                    ret[key] = np.mean(val)
            algo_to_score[name + str(i)]['cross_validation'] = ret
    pprint(algo_to_score)

if __name__ == '__main__':
    main()