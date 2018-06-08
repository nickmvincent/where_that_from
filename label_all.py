import glob
from label_sentences import label_sentences
from sklearn.model_selection import KFold
import pandas as pd

# TODO
sentences_filepaths = glob.glob("sentences/psa_research/*.csv")
for sentences_filepath in sentences_filepaths:
    print(sentences_filepath)
    label_sentences(sentences_filepath, mode='auto')

sentences_filepaths = glob.glob("labeled_sentences/*/*.csv")
data = None
for path in sentences_filepaths:
    if data is None:
        data = pd.read_csv(path, encoding='utf-8')
    else:
        data = pd.concat([data, pd.read_csv(path, encoding='utf-8')])
data.to_csv('all_labeled_sentences.csv')

fold = KFold(n_splits=5, shuffle=True, random_state=0)
for i, (train_index, test_index) in enumerate(fold.split(data, data.has_citation)):
    data.iloc[test_index].to_csv('pre_split_data/test_{}.csv'.format(i), encoding='utf8')
    train_df = data.iloc[train_index]
    
    outstr = ''
    for _, row in train_df.iterrows():
        outstr += row[0] + '``' + str(int(row['has_citation'])) + ' '
    
    with open('pre_split_data/train_{}.txt'.format(i), 'w', encoding='utf8') as f:
        f.write(outstr)
                