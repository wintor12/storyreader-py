import torch
import codecs
import torchtext.data
import torchtext.vocab


PAD_WORD = "<PAD>"


class StoryDataset(torchtext.data.Dataset):
    def __init__(self, fields, src_path, question_path,
                 feature_path=None, tgt_path=None, opt=None, **kwargs):
        examples = []
        src_words = []
        q_words = []
        self.src_vocabs = []
        with codecs.open(src_path, 'r', 'utf-8') as src_file, codecs.open(question_path, 'r', 'utf-8') as q_file:
            for i, (src_line, q_line) in enumerate(zip(src_file, q_file)):
                src = src_line.strip().split()
                question = q_line.strip().split()
                d = {'src': src, 'question': question, 'indices': i}
                examples.append(d)
                src_words.append(src)
                q_words.append(question)
        print(len(src_words))
        print(len(q_words))
        keys = examples[0].keys()
        fields = [(k, fields[k]) for k in keys]
        examples = list([torchtext.data.Example.fromlist([ex[k] for k in keys], fields)
                         for ex in examples])

        super(StoryDataset, self).__init__(examples, fields)

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
        return fields

    @staticmethod
    def build_vocab(train, opt):
        fields = train.fields
        fields['src'].build_vocab(train, max_size=20000)


def main():
    fields = StoryDataset.get_fields()
    train = StoryDataset(fields, 'data/s_train', 'data/q_train')
    print(train[0].src, train[0].question, train[0].indices)


if __name__ == "__main__":
    main()
