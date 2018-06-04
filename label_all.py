import glob
from label_sentences import label_sentences

sentences_filepaths = glob.glob("sentences/*.csv")
for sentences_filepath in sentences_filepaths:
    print(sentences_filepath)
    label_sentences(sentences_filepath, mode='auto')

rnn_sentences_filepaths = glob.glob("rnn_sentences/*.txt")
out = ''
for rnn_sentences_filepath in rnn_sentences_filepaths:
    with open(rnn_sentences_filepath, 'r') as f:
        text = f.read()
        out += text
with open('rnn.txt', 'w', encoding='utf8') as f:
    f.write(out)

