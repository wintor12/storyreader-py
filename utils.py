from tqdm import trange
import array
import six
import torch


def weight_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def print_weight_grad(model, opt):
    print('model', weight_grad_norm(model.parameters()))
    print('embed', weight_grad_norm(model.embed.parameters()))
    print('question', weight_grad_norm(model.q_rcnn.parameters()))
    print('story', weight_grad_norm(model.s_rcnn.parameters()))
    print('fc', weight_grad_norm(model.fc.parameters()))
    if opt.reader == 's':
        print('rnn_cell', weight_grad_norm(model.rnn_cell.parameters()))
        print('r_w', weight_grad_norm(model.r_w.parameters()))
        print('h_w', weight_grad_norm(model.h_w.parameters()))
    if opt.reader == 'h':
        print('rnn', weight_grad_norm(model.rnn.parameters()))
        print('h_conv', weight_grad_norm(model.h_conv.parameters()))
        print('r_conv', weight_grad_norm(model.r_conv.parameters()))


def load_word_vectors(path, wv_size, vocab, unk_init='random'):
    """Load word vectors from a path"""
    print('Loading word vectors from ' + path)
    cm = open(path, 'rb')
    cm = [line for line in cm]
    wv_tokens, wv_arr, wv_size = [], array.array('d'), None

    if cm is not None:
        for line in trange(len(cm)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue

            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)

    wv = torch.Tensor(len(vocab), wv_size)
    wv.normal_(0, 1) if unk_init == 'random' else wv.zero_()
    for i, token in enumerate(vocab.itos):
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            wv[i] = wv_arr[wv_index]
    return wv


class Statistics:

    def __init__(self, **kwargs):
        for key, value in list(kwargs.items()):
            if value is not None:
                setattr(self, key, value)
