import glob
from label_sentences import label_sentences

sentences_filepaths = glob.glob("sentences/*.csv")
for sentences_filepath in sentences_filepaths:
    print(sentences_filepath)
    label_sentences(sentences_filepath, mode='auto')
