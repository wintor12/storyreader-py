import torch
import codecs
import torchtext.data
import torchtext.vocab


PAD_WORD = "<pad>"


class StoryDataset(torchtext.data.Dataset):

    def __init__(self, fields, src_path, question_path,
                 feature_path, tgt_path, opt=None, **kwargs):
        examples = []
        self.src_vocabs = []
        with codecs.open(src_path, 'r', 'utf-8') as src_file, \
             codecs.open(question_path, 'r', 'utf-8') as q_file, \
             codecs.open(feature_path, 'r', 'utf-8') as f_file, \
             codecs.open(tgt_path, 'r', 'utf-8') as t_file:
            for i, (src_line, q_line, f_line, t_line) in enumerate(
                    zip(src_file, q_file, f_file, t_file)):

                src = src_line.strip().split()
                question = q_line.strip().split()
                feature = f_line.strip().split()
                feature = [float(x) for x in feature]
                tgt = float(t_line.strip())
                d = {'src': src, 'question': question, 'indices': i,
                     'feature': feature, 'tgt': tgt}
                examples.append(d)

        keys = examples[0].keys()
        fields = [(k, fields[k]) for k in keys]
        examples = list([torchtext.data.Example.fromlist([ex[k] for k in keys], fields)
                         for ex in examples])

        super(StoryDataset, self).__init__(examples, fields)

    @staticmethod
    def sort_key(ex):
        "Sort in reverse size order"
        return -len(ex.src)

    @staticmethod
    def get_fields():
        fields = {}
        fields['src'] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

        # question and src text share the same field
        fields['question'] = fields['src']

        fields['indices'] = torchtext.data.Field(
            use_vocab=False,
            tensor_type=torch.LongTensor,
            sequential=False)

        fields['feature'] = torchtext.data.Field(
            use_vocab=False,
            tensor_type=torch.FloatTensor,
            sequential=False)

        fields['tgt'] = torchtext.data.Field(
            use_vocab=False,
            tensor_type=torch.FloatTensor,
            sequential=False)

        return fields

    @staticmethod
    def build_vocab(train, opt):
        fields = train.fields
        fields['src'].build_vocab(train, max_size=opt.src_vocab_size)

    def __getstate__(self):
        "Need this for pickle save"
        return self.__dict__

    def __setstate__(self, d):
        "Need this for pickle load"
        self.__dict__.update(d)
