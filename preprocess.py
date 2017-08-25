import torch
import argparse
from dataset import StoryDataset
import dill
import utils


parser = argparse.ArgumentParser()

parser.add_argument('--text', action='store_true',
                    help='Use text feature only')

parser.add_argument('--train_src', default='s_train', help='train story texts')
parser.add_argument('--train_tgt', default='p_y_train', help='train upvotes')
parser.add_argument('--train_feature', default='p_feature_train',
                    help='train feature')
parser.add_argument('--train_question', default='q_train', help='train questions')
parser.add_argument('--valid_src', default='s_val', help='valid story texts')
parser.add_argument('--valid_tgt', default='p_y_val', help='valid upvotes')
parser.add_argument('--valid_feature', default='p_feature_val',
                    help='valid feature')
parser.add_argument('--valid_question', default='q_val', help='valid questions')
parser.add_argument('--test_src', default='s_test', help='test story texts')
parser.add_argument('--test_tgt', default='p_y_test', help='test upvotes')
parser.add_argument('--test_feature', default='p_feature_test',
                    help='test feature')
parser.add_argument('--test_question', default='q_test', help='test questions')

parser.add_argument('--data', default='./data/', help='output file for the prepared data')

parser.add_argument('--src_vocab_size', type=int, default=20000,
                    help='size of the source vocabulary')
parser.add_argument('--fix_length', type=int, default=360,
                    help='fix the length of the story, if 0, flexible length')
parser.add_argument('--stopwords', action='store_true',
                    help='if true, remove stopwords')

parser.add_argument('--word_vec_size', type=int, default=300,
                    help='Word embedding sizes')
parser.add_argument('--pre_word_vec', type=str, default='',
                    help='pre-trained Word embedding path')
parser.add_argument('--word_vec_only', action='store_true',
                    help='Only preprocess word embeddings')


opt = parser.parse_args()
print(opt)


def main():
    print('Preprocessing ... ')
    if opt.text:
        opt.train_tgt = 'diff_y_train'
        opt.valid_tgt = 'diff_y_val'
        opt.test_tgt = 'diff_y_test'
    
    fields = StoryDataset.get_fields(opt)
    train = StoryDataset(fields, opt.data + opt.train_src,
                         opt.data + opt.train_question,
                         opt.data + opt.train_feature,
                         opt.data + opt.train_tgt, opt)
    valid = StoryDataset(fields, opt.data + opt.valid_src,
                         opt.data + opt.valid_question,
                         opt.data + opt.valid_feature,
                         opt.data + opt.valid_tgt, opt)
    test = StoryDataset(fields, opt.data + opt.test_src,
                        opt.data + opt.test_question,
                        opt.data + opt.test_feature,
                        opt.data + opt.test_tgt, opt)

    print('Building Vocab ... ')
    StoryDataset.build_vocab(train, opt)

    if opt.pre_word_vec:
        print('Saving pretrained word vectors ... ')
        wv = utils.load_word_vectors(opt.pre_word_vec, opt.word_vec_size,
                                     fields['src'].vocab, unk_init='random')
        torch.save(wv, opt.data + 'wv.pt', pickle_module=dill)
        if opt.word_vec_only:
            return

    print('Saving train ... ')
    torch.save(train, opt.data + 'train.pt', pickle_module=dill)
    print('Saving valid ...')
    torch.save(valid, opt.data + 'valid.pt', pickle_module=dill)
    print('Saving test ...')
    torch.save(test, opt.data + 'test.pt', pickle_module=dill)
    print('Saving fields ...')
    torch.save(fields, opt.data + 'fields.pt', pickle_module=dill)


if __name__ == "__main__":
    main()
