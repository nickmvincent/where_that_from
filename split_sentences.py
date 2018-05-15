from nltk import tokenize

def test():
    """Driver"""
    filename = 'test.txt'
    with open(filename, 'r') as f:
        data = f.read()
    data = data.replace('Fig.', 'Figure')
    sentences = tokenize.sent_tokenize(data)
    print(sentences)
    # erroneously splits on "Fig. 3"
    assert len(sentences) == 6


def split_sentences(filename='sample_paper.txt'):
    """Driver"""
    with open(filename, 'r') as f:
        data = f.read()
    pairs = {
        'Fig': 'Fig',
        'e.g.': 'eg',
        'i.e.': 'ie',
        'et al.': 'et al',
    }
    for key, val in pairs.items():
        data = data.replace(key, val)
    sentences = tokenize.sent_tokenize(data)
    print(sentences)

    out = ['text,label']
    for sentence in sentences:
        out.append('`' + sentence + '`,nan')
    with open('sentences/{}'.format(
        filename.replace('.txt', '.csv')
    ), 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    

if __name__ == '__main__':
    split_sentences()