from collections import Counter, OrderedDict

class Vocab:
    """
    A custom vocabulary class.
    It maps tokens to indices and vice-versa.
    """
    def __init__(self, ordered_dict, specials):
        self.specials = specials
        self.stoi = {token: i for i, token in enumerate(specials)}
        # Start indexing words from the end of special tokens
        idx_offset = len(specials)
        for i, (token, _) in enumerate(ordered_dict.items()):
            self.stoi[token] = i + idx_offset
        
        self.itos = {i: token for token, i in self.stoi.items()}
        self._default_index = self.stoi['<unk>']

    def __getitem__(self, token):
        return self.stoi.get(token, self._default_index)

    def __len__(self):
        return len(self.stoi)
        
    def get_itos(self):
        # Return a list of tokens sorted by index
        return [self.itos[i] for i in range(len(self.itos))]
