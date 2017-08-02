import torch
import argparse
from dataset import StoryDataset
import dill


parser = argparse.ArgumentParser()
parser.add_argument('--train_src', default='./data/s_train', help='train story texts')
parser.add_argument('--train_tgt', default='./data/y_train', help='train upvotes')
parser.add_argument('--train_feature', default='./data/feature_train',
                    help='train feature')
parser.add_argument('--train_question', default='./data/q_train', help='train questions')
parser.add_argument('--valid_src', default='./data/s_val', help='valid story texts')
parser.add_argument('--valid_tgt', default='./data/y_val', help='valid upvotes')
parser.add_argument('--valid_feature', default='./data/feature_val', help='valid feature')
parser.add_argument('--valid_question', default='./data/q_val', help='valid questions')
parser.add_argument('--save', default='./data/', help='output file for the prepared data')

parser.add_argument('--src_vocab_size', type=int, default=20000,
                    help='size of the source vocabulary')

opt = parser.parse_args()
print(opt)


def main():
    print('Preprocessing ... ')

    fields = StoryDataset.get_fields()
    train = StoryDataset(fields, opt.train_src, opt.train_question,
                         opt.train_feature, opt.train_tgt)
    valid = StoryDataset(fields, opt.valid_src, opt.valid_question,
                         opt.valid_feature, opt.valid_tgt)

    print('Building Vocab ... ')
    StoryDataset.build_vocab(train, opt)

    print('Saving train ... ')
    torch.save(train, opt.save + 'train.pt', pickle_module=dill)
    print('Saving valid ...')
    torch.save(valid, opt.save + 'valid.pt', pickle_module=dill)
    print('Saving fields ...')
    torch.save(fields, opt.save + 'fields.pt', pickle_module=dill)


if __name__ == "__main__":
    main()
