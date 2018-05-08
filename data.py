import os

def read_conll_sentence(path, word_vocab, label_vocab, fields=('word', 'pos_tag', 'chunk', 'bio')):
    sent = [[] for _ in range(len(fields))]
    with open(path) as f:
        for line in f:
            line = line.strip().split()
            if line:
                i = 0
                if 'word' in fields:
                    sent[i].append(word_vocab[line[0]])
                    i += 1
                if 'pos_tag' in fields:
                    sent[i].append(line[1])
                    i += 1
                if 'chunk' in fields:
                    sent[i].append(line[2])
                    i += 1
                if 'bio' in fields:
                    sent[i].append(label_vocab[line[3]])
            else:
                yield tuple(sent)
                sent = [[] for _ in range(len(fields))]


def write_conll(path, data):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        for sent in data:
            for word in sent:
                print(' '.join(word), file=f)
            print('', file=f)


# def read_conll_spanish_sent(path, word_vocab, label_vocab, fields=('word', 'pos_tag', 'bio')):
#     sent = [[] for _ in range(len(fields))]
#     with open(path, 'r', encoding='iso-8859-1') as f:
#         for line in f:
#             line = line.strip().split()
#             if line:
#                 i = 0
#                 if 'word' in fields:
#                     sent[i].append(word_vocab[line[0]])
#                     i += 1
#                 if 'pos_tag' in fields:
#                     sent[i].append(label_vocab[line[1]])
#                     i += 1
#                 if 'bio' in fields:
#                     sent[i].append(line[2])
#             else:
#                 yield tuple(sent)
#                 sent = [[] for _ in range(len(fields))]


def read_conll_spanish_sent(path, word_vocab, label_vocab, fields=('word', 'pos_tag')):
    sent = [[] for _ in range(len(fields))]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip().split()
            if line:
                i = 0
                if 'word' in fields:
                    sent[i].append(word_vocab[line[2]])
                    i += 1
                if 'pos_tag' in fields:
                    sent[i].append(label_vocab[line[3]])
                    i += 1
            else:
                yield tuple(sent)
                sent = [[] for _ in range(len(fields))]
