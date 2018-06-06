from nltk import tokenize
import re

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


def split_sentences(filepath='sample_paper.txt'):
    """Driver"""
    with open(filepath, 'r') as f:
        data = f.read()
    
    # chop off the references
    ref_matches = [m.start() for m in re.finditer('references 1', data.lower())]
    if ref_matches:
        data = data[:ref_matches[-1]]
    else:
        print('Did not find references 1 so let us try just finding the word references')
        ref_matches = [m.start() for m in re.finditer('references', data.lower())]
        if ref_matches:
            data = data[:ref_matches[-1]]
        else:
            raise ValueError('No references in {}'.format(filepath))
        
    
    # do some replacement
    pairs = {
        'Fig.': 'Fig',
        'e.g.': 'eg',
        'i.e.': 'ie',
        'et al.': 'et al',
    }
    for key, val in pairs.items():
        data = data.replace(key, val)

    sentences = tokenize.sent_tokenize(data)
    #print(sentences)

    out = ['text,label,has_citation']
    for sentence in sentences:
        out.append('"' + sentence.replace('"', '""') + '",nan,nan')
    csv_filepath = filepath.replace('txt_files', 'sentences').replace('.txt', '.csv')
    with open(csv_filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    

if __name__ == '__main__':
    split_sentences()