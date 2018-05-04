from nltk import tokenize

def main():
    """Driver"""
    filename = 'test.txt'
    with open(filename, 'r') as f:
        data = f.read()
    data = data.replace('Fig.', 'Figure')
    sentences = tokenize.sent_tokenize(data)
    print(sentences)
    # erroneously splits on "Fig. 3"
    assert len(sentences) == 6

    out = ['text,label']
    for sentence in sentences:
        out.append('"' + sentence + '",0')
    with open('sentences.csv', 'w') as f:
        f.write('\n'.join(out))

if __name__ == '__main__':
    main()