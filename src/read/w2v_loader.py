from read.vocabulary import Vocabulary
import numpy as np
import os


class Model(object):
    def __init__(self, vocab, words, embeddings):
        self.embeddings = embeddings
        self.vocab = vocab
        self.words = words
        self.dim = len(embeddings[0])
        self.length = len(embeddings)


def load_vectors(filename, vocab=None):
    word2idx = {}
    words = []
    with open(filename, 'r') as f:
        first_line = f.readline()
        vocab_no, dim = map(int, first_line.split())
        print(vocab_no, dim)

        if vocab:
            vocab_no = len(vocab)

        lookup = np.empty([vocab_no, dim], dtype=np.float)
        f.seek(0)
        n = 0
        idx = 0
        for line in f:
            if idx == 0:
                idx += 1
            else:
                split = line.strip().split(maxsplit=1)
                word = split[0]
                vec = split[1]

                if vocab is None or word in vocab:
                    word2idx[word] = idx - 1
                    words.append(word)
                    v = np.fromstring(vec, sep=' ')
                    if len(v) == dim:
                        lookup[idx - 1] = v
                        idx += 1
                    else:
                        print(line)
                n += 1
                if n % 10000 == 0:
                    print('  {}th vector done..'.format(n))
    lookup = lookup[0:idx - 1]
    lookup.resize([idx - 1, dim])

    return_vocab = Vocabulary(word2idx)
    return Model(return_vocab, words, lookup)

#
# if __name__ == "__main__":
#     vocab, lookup = load_vectors('/Users/marziehsaeidi/Documents/Apps/UrbanScala/data/Regression/features/w2vy.txt')
#     print(vocab)
#     print(lookup)
