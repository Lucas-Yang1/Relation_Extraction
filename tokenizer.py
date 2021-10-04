from transformers import BertTokenizer


class Tokenizer(BertTokenizer):

    def _tokenize(self, text):
        split_tokens = []
        for c in text:
            if c in self.vocab:
                split_tokens.append(c)
            elif c == ' ':
                split_tokens.append('[unused1]')
            else:
                split_tokens.append('[UNK]')

        return split_tokens

