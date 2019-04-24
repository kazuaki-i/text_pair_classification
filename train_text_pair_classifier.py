import argparse
import datetime
import json
import os
import time

import chainer
from chainer import training
from chainer.training import extensions, triggers

import nets
from nlp_utils import *


def main():
    start = time.time()
    current_datetime = '{}'.format(datetime.datetime.today())
    parser = argparse.ArgumentParser(description='Chainer Text Classification')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=200,
                        help='Number of units')
    parser.add_argument('--vocab', '-v', type=int, default=100000,
                        help='Number of max vocabulary')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='Number of layers of RNN or MLP following CNN')
    parser.add_argument('--dropout', '-d', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--dataset', '-dataset', required=True, help='train dataset')
    parser.add_argument('--size', '-size', type=int, default=-1,
                        help='train dataset size -> def train:3/4, test:1/4')
    parser.add_argument('--model', '-model', default='cnn',
                        choices=['cnn', 'lstm', 'bow', 'gru'],
                        help='Name of encoder model type.')
    parser.add_argument('--early-stop', action='store_true', help='use early stopping method')
    parser.add_argument('--same-network', action='store_true', help='use same network between i1 and i2')
    parser.add_argument('--save-init', action='store_true', help='save init model')
    parser.add_argument('--char-based', action='store_true')

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    train, test, vocab = get_input_dataset(args.dataset, vocab=None, max_vocab_size=args.vocab)

    print('# train data: {}'.format(len(train)))
    print('# dev  data: {}'.format(len(test)))
    print('# vocab: {}'.format(len(vocab)))
    n_class = len(set([int(d[-1]) for d in train]))
    print('# class: {}'.format(n_class))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # Setup a model
    if args.model == 'lstm':
        Encoder = nets.LSTMEncoder
    elif args.model == 'cnn':
        Encoder = nets.CNNEncoder
    elif args.model == 'bow':
        Encoder = nets.BOWMLPEncoder
    elif args.model == 'gru':
        Encoder = nets.GRUEncoder

    encoder = Encoder(n_layers=args.layer, n_vocab=len(vocab), n_units=args.unit,
                      dropout=args.dropout, same_network=args.same_network)
    model = nets.TextClassifier(encoder, n_class)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    # Set up a trainer
    updater = training.updaters.StandardUpdater(train_iter, optimizer, converter=convert_seq2, device=args.gpu)

    # early Stopping
    if args.early_stop:
        stop_trigger = triggers.EarlyStoppingTrigger(monitor='validation/main/loss', max_trigger=(args.epoch, 'epoch'))
    else:
        stop_trigger = (args.epoch, 'epoch')

    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, converter=convert_seq2, device=args.gpu))

    # Take a best snapshot
    record_trigger = training.triggers.MaxValueTrigger('validation/main/accuracy', (1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'best_model.npz'), trigger=record_trigger)

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    # trainer.extend(extensions.ProgressBar())

    # Save vocabulary and model's setting
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    vocab_path = os.path.join(args.out, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    model_path = os.path.join(args.out, 'best_model.npz')
    model_setup = args.__dict__
    model_setup['vocab_path'] = vocab_path
    model_setup['model_path'] = model_path
    model_setup['n_class'] = n_class
    model_setup['datetime'] = current_datetime
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)

    if args.save_init:
        chainer.serializers.save_npz(os.path.join(args.out, 'init_model.npz'), model)
        exit()

    # Run the training
    print('Start trainer.run: {}'.format(current_datetime))
    trainer.run()
    print('Elapsed_time: {}'.format(datetime.timedelta(seconds=time.time()-start)))


if __name__ == '__main__':
    main()
