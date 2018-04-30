
def read_conll_sentence(path, word_vocab, label_vocab):
    sent = []
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip().split()
            if line:
                sent.append(word_vocab[line[0]])
                labels.append(label_vocab[line[3]])
            else:
                yield sent, labels
                sent = []
                labels = []


def write_conll(path, data):
    with open(path, 'w') as f:
        for sent in data:
            for word in sent:
                print(' '.join(word), file=f)
            print('', file=f)
