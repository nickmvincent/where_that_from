"""
try to classify reddit posts.
"""
import os
import glob
from collections import defaultdict
from pprint import pprint
import time
from datetime import datetime

import pandas as pd
from sklearn_pandas import DataFrameMapper, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import cross_validate, KFold, train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier



def print_topk(k, feature_names, clf):
    """Prints features with the highest coefficient values, per class"""
    topk = np.argsort(clf.coef_[0])[-k:]
    print(
        "{}".format(
            " ".join(feature_names[j] for j in topk[::-1])
        )
    )

SCORES = [
    'accuracy', 
    'roc_auc', 
    #'recall',
    #'precision',
]


def run_experiment(X, y, max_features, feature_selector, args):
    
    long_precision_recall_row_dicts = []
    long_result_row_dicts = []
    algo_to_score = defaultdict(dict)
    clf_sets = []
    C_vals = [0.01, 0.1, 1, 10, 100]
    
    for C_val in C_vals:
        clf_sets += [
            (LogisticRegression(C=C_val), 'logistic__c={}'.format(C_val), 'logistic', C_val, 'default'),
        ]
        clf_sets += [
            (LinearSVC(C=C_val), 'linearsvc__c={}'.format(C_val), 'linearsvc', C_val, 'default'),
        ]

        if args.other_class_weights:
            clf_sets += [
                (
                    LogisticRegression(C=C_val, class_weight='balanced'),
                    'logistic__c={}__class_weight=balanced'.format(C_val),
                    'logistic',
                    C_val, 'balanced'
                ),
                (
                    LogisticRegression(C=C_val, class_weight={0: 1, 1:50}), 'logistic__c={}__class_weight=50x'.format(C_val),
                    'logistic',
                    C_val, '50x'
                ),
                (
                    LinearSVC(C=C_val, class_weight='balanced'), 'linearsvc__c={}__class_weight=balanced'.format(C_val),
                    'linearsvc',
                    C_val, 'balanced'
                ),
                (
                    LinearSVC(C=C_val, class_weight={0: 1, 1:50}), 'linearsvc__c={}__class_weight=500x'.format(C_val),
                    'linearsvc',
                    C_val, '50x'
                ),
            ]
    
    clf_sets += [
        (DummyClassifier(strategy='most_frequent'), 'SelectNoSentences', 'SelectNoSentences', 0.1, 'default'),
        #(DummyClassifier(strategy='constant', constant=1), 'SelectEverySentence',),
        (DecisionTreeClassifier(), 'tree', 'tree', 0.1, 'default'),
        (GaussianNB(), 'GaussianNb', 'GaussianNb', 0.1, 'default'),
        #(MultinomialNB(), 'MultinomialNB', 'MultinomialNB', 0.1, 'default'),
        (KNeighborsClassifier(3), '3nn', '3nn', 0.1, 'default'),
    ]
    for clf, name, algo_name, C_val, weights in clf_sets:
        # if name == 'logistic':
        #     clf.fit(X, data.has_citation)
        #     print_topk(10, mapper.transformed_names_, clf)

        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        start = time.time()
        scores = cross_validate(
            clf, X=X, y=y, cv=cv,
            scoring=SCORES)
        ret = {}
        for key, val in scores.items():
            if 'test_' in key:
                ret[key.replace('test_', '')] = np.mean(val)
        algo_to_score[name] = ret
        tic = round(time.time() - start, 3)
        algo_to_score[name]['time'] = tic
        

        X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=0)
        cal_clf = CalibratedClassifierCV(clf) 
        cal_clf.fit(X_train, y_train)
        y_proba = cal_clf.predict_proba(X_test)
        precision, recall, thresholds = precision_recall_curve(
            y_test, y_proba[:,1])
        
        long_result_row_dict = {
            'name': name,
            'algo_name': algo_name,
            'max_features': max_features,
            'feature_selector': feature_selector.__name__,
            'C_val': C_val,
            'weights': weights,
            'time': tic,
        }
        for score in SCORES:
            long_result_row_dict[score] = algo_to_score[name][score]
        long_result_row_dicts.append(long_result_row_dict)

        for precision_val, recall_val in zip(precision, recall):
            long_precision_recall_row_dicts.append({
                'precision': precision_val,
                'recall': recall_val,
                'name': name,
                'max_features': max_features,
                'feature_selector': feature_selector.__name__,
                'algo_name': algo_name,
                'C_val': C_val,
                'weights': weights,
            })
        #print(name, tic)
    result_df = pd.DataFrame(algo_to_score)
    result_df.to_csv('results/{}/{}_{}.csv'.format(
        args.data_dir,
        max_features,
        feature_selector.__name__,
    ))
    maxes = {}

    for key in SCORES:
        max_idx = result_df.loc[key].idxmax()
        max_val = result_df.loc[key, max_idx]
        maxes[key] = [max_val, max_idx]
    #print(maxes)

    long_results_df = pd.DataFrame(long_result_row_dicts)
    long_precision_recall_df = pd.DataFrame(long_precision_recall_row_dicts)
    # sns.factorplot(data=long_df, x='recall', y='precision', col='name', col_wrap=5)
    # plt.show()

    return maxes, long_results_df, long_precision_recall_df

def main(args):
    if False:
        path = 'labeled_sentences/vincent_etal_chi_2018.csv'
        data = pd.read_csv(path, encoding='utf-8t')
    else:
        sentences_filepaths = glob.glob(
            "labeled_sentences/{}/*.csv".format(args.data_dir)
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

    # "Feature Engineering"

    def compute_hand_features(processed_text):
        ret = [
            len(processed_text),
            any(char.isdigit() for char in processed_text),
             ',' in processed_text,
             '"' in processed_text,
             any(char.isupper() for char in processed_text[1:]),
        ]
        return ret

    data['length'], data['has_digits'], data['has_comma'], data['has_quote'], data['has_upper'] = zip(
        *map(compute_hand_features, data['processed_text'])
    )
    mapper = DataFrameMapper([
        ('processed_text', 
            TfidfVectorizer(
                #stop_words='english', 
                #lowercase=True,
                #ngram_range=(5,5)
                strip_accents='unicode',
                #max_features=10000
                )
        ),
        (['length'], StandardScaler()),
        ('has_digits', LabelBinarizer()),
        ('has_comma', LabelBinarizer()),
        ('has_quote', LabelBinarizer()),
        ('has_upper', LabelBinarizer()),
    ])
    X = mapper.fit_transform(data.copy())
    y = data.has_citation

    feature_selectors = [
        f_classif,
        #mutual_info_classif,
    ]
    maxes = {x: [0, None] for x in SCORES}
    long_result_dfs = []
    long_precision_recall_dfs = []
    #print(maxes)
    best_feature_max = None
    best_feature_selector = None
    for max_features in args.max_features_vals:
        for feature_selector in feature_selectors:
            if max_features != -1:
                #print('Features:', max_features)
                X_new = SelectKBest(feature_selector, k=max_features).fit_transform(X, y)

            if args.run:
                maxes_from_experiment, long_result_df, long_precision_recall_df = run_experiment(X_new, y, max_features, feature_selector, args)
                long_precision_recall_dfs.append(long_precision_recall_df)
                long_result_dfs.append(long_result_df)
                for score in SCORES:
                    if maxes_from_experiment[score][0] > maxes[score][0]:
                        maxes[score] = maxes_from_experiment[score] + ['{}_{}'.format(max_features, feature_selector.__name__)]
                        if score == 'roc_auc':
                            best_feature_max = max_features
                            best_feature_selector = feature_selector
            if args.grid:
                # Split the dataset in two equal parts
                X_train, X_test, y_train, y_test = train_test_split(
                    X_new, y, test_size=0.5, random_state=0)
                tuned_parameters = [
                    {
                        'C': [0.1, 1, 10, 100, 1000]
                    }
                ]
                scores = ['precision', 'recall']
                for score in scores:
                    print("# Tuning hyper-parameters for %s" % score)
                    print()

                    clf = GridSearchCV(LinearSVC(), tuned_parameters, cv=5,
                                    scoring='%s_macro' % score)
                    clf.fit(X_train, y_train)

                    print("Best parameters set found on development set:")
                    print(clf.best_params_)
                    print("Grid scores on development set:")
                    means = clf.cv_results_['mean_test_score']
                    stds = clf.cv_results_['std_test_score']
                    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                        print("%0.3f (+/-%0.03f) for %r"
                            % (mean, std * 2, params))
                    print("Detailed classification report:")
                    print("The model is trained on the full development set.")
                    print("The scores are computed on the full evaluation set.")
                    y_true, y_pred = y_test, clf.predict(X_test)
                    print(classification_report(y_true, y_pred))
    
    print(maxes)

    merged_precision_recall_df = long_precision_recall_dfs[0]
    for long_df in long_precision_recall_dfs[1:]:
        merged_precision_recall_df = pd.concat([merged_precision_recall_df, long_df])
    merged_precision_recall_df.to_csv('results/{}/precision_recall_dataframe.csv'.format(args.data_dir))

    merged_result_df = long_result_dfs[0]
    for long_df in long_result_dfs[1:]:
        merged_result_df = pd.concat([merged_result_df, long_df])
    merged_result_df.to_csv('results/{}/result_dataframe.csv'.format(args.data_dir))
    if args.prc:
        if best_feature_selector is None:
            # set this manually I suppose
            best_feature_selector = f_classif
            best_feature_max = 100
        X_new = SelectKBest(best_feature_selector, k=best_feature_max).fit_transform(X, y)    
        X_train, X_test, y_train, y_test = train_test_split(
                    X_new, y, test_size=0.2, random_state=0)
        svm = LinearSVC(C=0.1)
        clf = CalibratedClassifierCV(svm) 
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)
        precision, recall, thresholds = precision_recall_curve(
            y_test, y_proba[:,1])
        plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                        color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve')
        plt.show()
        
def parse():
    import argparse

    parser = argparse.ArgumentParser(description='Classify')
    parser.add_argument('--grid', action='store_true', help='Do grid search')
    parser.add_argument('--run', action='store_true',
                        help='run experiments')
    parser.add_argument('--prc', action='store_true',
                        help='Draw precision recall curve')
    parser.add_argument('--max_features_vals', default='50,100,500,1000,2000,3000,4000,5000',
                        help='max # of features to use')
    parser.add_argument('--other_class_weights', action='store_true', help='try alternate class weights')
    parser.add_argument('--data_dir', default='psa_research',
                        help='Where is the data?')
    args = parser.parse_args()

    args.max_features_vals = [int(x) for x in args.max_features_vals.split(',')] 
    if False:
        args.max_features_vals += ['all']

    main(args)

if __name__ == '__main__':
    parse()