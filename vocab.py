from collections import defaultdict


class Vocabulary:
    def __init__(self, unk_symbol, sos_symbol=None, eos_symbol=None):
        # NOTE: keep these two updated with each other
        self.unk_symbol = unk_symbol
        self.w2i = defaultdict(lambda: len(self.w2i))
        self.i2w = dict()
        self.is_frozen = False
        self.unk_idx = self.__getitem__(unk_symbol)  # add to 0th index
        if sos_symbol:
            self.sos_symbol = sos_symbol
            self.sos_idx = self.__getitem__(sos_symbol)
        if eos_symbol:
            self.eos = eos_symbol
            self.eos_idx = self.__getitem__(eos_symbol)

    def __getitem__(self, word):
        """
        Get the index of the word in the vocab. If it is not in the vocab and the vocab is not frozen,
        add it to the vocab.
        """
        if self.is_frozen:
            if word in self.w2i:
                return self.w2i[word]
            return self.unk_idx
        index = self.w2i[word]
        self.i2w[index] = word
        return index

    def __len__(self):
        return len(self.w2i)

    def freeze(self):
        self.is_frozen = True


if __name__ == '__main__':
    print('Testing...')
    v = Vocabulary('<unk>')
    assert(len(v) == 1)
    assert(v['hello'] == 1)
    assert(len(v) == 2)
    assert(list(v.w2i.keys()) == ['<unk>', 'hello'])
    assert(list(v.i2w.keys()) == [0, 1])
    v.freeze()
    assert(v['hello'] == 1)
    assert(v['hi'] == 0)
    assert(len(v) == 2)
    print('Done.')
