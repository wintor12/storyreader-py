from tqdm import trange
import array
import six
import torch


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
