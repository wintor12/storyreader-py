import torch
import codecs
import torchtext.data
import torchtext.vocab
import numpy as np
from nltk.corpus import stopwords


PAD_WORD = "<pad>"


class StoryDataset(torchtext.data.Dataset):

    def __init__(self, fields, src_path, question_path,
                 feature_path, tgt_path, opt, **kwargs):

        def src_preprocessing(src_list):
            # remove stopwords
            if opt.stopwords:
                stops = set(stopwords.words("english"))
                src_list = [word for word in src_list if word not in stops]
                if len(src_list) < 36:
                    src_list += [PAD_WORD] * (36 - len(src_list))

            processed_src = []
            if opt.fix_length > 0:
                fix_length_src = opt.fix_length
                if len(src_list) == fix_length_src:
                    return src_list

                chunked_src = np.array_split(np.array(src_list), 10)
                if len(src_list) < fix_length_src:
                    for c in chunked_src:
                        num_pad = fix_length_src / 10 - len(c)
                        c = c.tolist()
                        leading_pads = [PAD_WORD] * int(num_pad / 2)
                        trailing_pads = [PAD_WORD] * int(num_pad - len(leading_pads))
                        processed_src = leading_pads + c + trailing_pads
                else:
                    for c in chunked_src:
                        start = int((len(c) - fix_length_src / 10) / 2)
                        temp = c[start:int(start + fix_length_src / 10)].tolist()
                        processed_src += temp
            return processed_src if processed_src else src_list

        def question_preprocessing(q_list):
            # remove stopwords
            if opt.stopwords:
                stops = set(stopwords.words("english"))
                q_list = [word for word in q_list if word not in stops]
            q_list = q_list[:36]
            if len(q_list) < 36:
                q_list = q_list + [PAD_WORD] * (36 - len(q_list))
            return q_list

        examples = []
        self.src_vocabs = []
        with codecs.open(src_path, 'r', 'utf-8') as src_file, \
             codecs.open(question_path, 'r', 'utf-8') as q_file, \
             codecs.open(feature_path, 'r', 'utf-8') as f_file, \
             codecs.open(tgt_path, 'r', 'utf-8') as t_file:
            for i, (src_line, q_line, f_line, t_line) in enumerate(
                    zip(src_file, q_file, f_file, t_file)):

                src = src_line.strip().split()
                src = src_preprocessing(src)
                question = q_line.strip().split()
                question = question_preprocessing(question)
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
    def get_fields(opt):
        fields = {}
        fields['src'] = torchtext.data.Field(
            pad_token=PAD_WORD,
            fix_length=opt.fix_length if opt.fix_length > 0 else None,
            lower=True,
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
