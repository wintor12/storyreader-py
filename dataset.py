import torch
import codecs
import torchtext.data
import torchtext.vocab


class StoryDataset(torchtext.data.Dataset):
    def __init__(self, src_path, feature_path, tgt_path, opt, **kwargs):
        examples = []
        src_words = []
        self.src_vocabs = []
        with codecs.open(src_path, 'r', 'utf-8') as src_file:
            for i, src_line in enumerate(src_file):
                src = src_line.split()
                d = {'src': src, 'indices': i}
                examples.append(d)
                src_words.append(src)


    @staticmethod
    def get_fields(src_path):
        with codecs.open(src_path, 'r', 'utf-8') as src_file:
           src_line = src_file.readline


    @staticmethod
    def build_vocab(train, opt):
        fields = train.fields
        fields['src'].build_vocab(train, max_size=20000)
