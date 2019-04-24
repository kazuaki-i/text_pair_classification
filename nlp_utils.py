import collections
import io

import numpy

import chainer
from chainer.backends import cuda


def split_text(text, char_based=False):
    if char_based:
        return list(text)
    else:
        return text.split()


def normalize_text(text):
    return text.strip().lower()


def make_vocab(dataset, max_vocab_size=20000, min_freq=2):
    counts = collections.defaultdict(int)
    for tokens, _ in dataset:
        for token in tokens:
            counts[token] += 1

    vocab = {'<eos>': 0, '<unk>': 1}
    for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if len(vocab) >= max_vocab_size or c < min_freq:
            break
        vocab[w] = len(vocab)
    return vocab


def read_vocab_list(path, max_vocab_size=20000):
    vocab = {'<eos>': 0, '<unk>': 1}
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        for l in f:
            w = l.strip()
            if w not in vocab and w:
                vocab[w] = len(vocab)
            if len(vocab) >= max_vocab_size:
                break
    return vocab


def make_array(tokens, vocab, add_eos=True):
    unk_id = vocab['<unk>']
    eos_id = vocab['<eos>']
    ids = [vocab.get(token, unk_id) for token in tokens]
    if add_eos:
        ids.append(eos_id)
    return numpy.array(ids, numpy.int32)


def convert_seq2(batch, device=None, with_label=True):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x)
                                     for x in batch[:-1]], dtype=numpy.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    if with_label:
        return {'xs1': to_device_batch([x1 for x1, _, _ in batch]),
                'xs2': to_device_batch([x2 for _, x2, _ in batch]),
                'ys': to_device_batch([y for _, _, y in batch])}
    else:
        return to_device_batch([x for x in batch])


def make_vocab2(dataset, max_vocab_size, min_freq=2):
    counts = collections.defaultdict(int)
    for t1, t2, _ in dataset:
        tokens = t1 + t2
        for token in tokens:
            counts[token] += 1

    vocab = {'<eos>': 0, '<unk>': 1}
    for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if len(vocab) >= max_vocab_size or c < min_freq:
            break
        vocab[w] = len(vocab)
    return vocab


def transform_to_array2(dataset, vocab, with_label=True):
    if with_label:
        return [(make_array(t1, vocab), make_array(t2, vocab), numpy.array([cls], numpy.int32))
                for t1, t2, cls in dataset]
    else:
        return [(make_array(t1, vocab), make_array(t2, vocab)) for t1, t2 in dataset]


def load_input_file(fi_name):
    print('load {} file'.format(fi_name))
    rl = []
    with open(fi_name, encoding='utf-8') as fi:
        for line in fi:
            l_lst = line.strip().split('\t')
            if len(l_lst) < 3:
                continue
            i1, i2 = l_lst[0].split(' '), l_lst[1].split(' ')
            label = l_lst[2]

            rl.append((i1, i2, label,))

    return rl


def get_input_dataset(fi_name, vocab=None, max_vocab_size=100000):
    dataset = load_input_file(fi_name)
    size = len(dataset) // 4 * 3

    train, test = dataset[:size], dataset[size:]

    if vocab is None:
        print('constract vocabulary based on frequency')
        vocab = make_vocab2(train, max_vocab_size)

    train = transform_to_array2(train, vocab)
    test = transform_to_array2(test, vocab)

    return train, test, vocab
