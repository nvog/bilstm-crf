import argparse
import random
import numpy as np
import dynet as dy
from vocab import Vocabulary
import data
import datetime
import os
import time


UNK = '<UNK>'
START = '<START>'
STOP = '<STOP>'


class BiLSTMCRF:
    def __init__(self, embed_dim, hidden_dim, nlayers, word_vocab, label_vocab):
        self.pc_all = dy.ParameterCollection()
        self.pc = self.pc_all.add_subcollection('to-save')
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.embeddings = self.pc_all.add_lookup_parameters((len(word_vocab), embed_dim))
        self.bilstm = dy.BiRNNBuilder(nlayers, embed_dim, hidden_dim * 2, self.pc, dy.LSTMBuilder)
        self.W_output = self.pc.add_parameters((len(label_vocab), hidden_dim * 2))
        self.b_output = self.pc.add_parameters(len(label_vocab))
        # transitions[x][y] from y to x
        transitions = np.random.randn(len(label_vocab), len(label_vocab))
        transitions[self.label_vocab.sos_idx, :] = -10000.0  # don't allow transitions from a label to START
        transitions[:, self.label_vocab.eos_idx] = -10000.0  # don't allow transitions from END to a label
        self.transitions = self.pc.parameters_from_numpy(transitions)

    def extract_feats(self, sent):
        bilstm_outputs = self.bilstm.transduce([self.embeddings[word_idx] for word_idx in sent])
        feats = dy.affine_transform([dy.parameter(self.b_output), dy.parameter(self.W_output),
                                            dy.concatenate_cols(bilstm_outputs)])
        return dy.transpose(feats)

    def viterbi_decode(self, feats):
        scores = np.full((1, len(self.label_vocab)), -10000.0, dtype=float)
        scores[0, self.label_vocab.sos_idx] = 0.0
        backpointers = []
        transitions = dy.parameter(self.transitions)

        for feat in feats:
            backpointers_t = []
            scores_t = []
            for next_tag in range(len(self.label_vocab)):
                next_scores = scores + dy.transpose(transitions[next_tag]).npvalue()
                best_tag_idx = np.argmax(next_scores)
                backpointers_t.append(best_tag_idx)
                scores_t.append(next_scores[0, best_tag_idx])
            scores = np.array(scores_t, ndmin=2) + dy.transpose(feat).npvalue()
            backpointers.append(backpointers_t)

        final_scores = scores + dy.transpose(transitions[self.label_vocab.eos_idx]).npvalue()
        best_tag_idx = np.argmax(final_scores)
        score = final_scores[0, best_tag_idx]

        # Trace backpointers to recover path
        best_path = [best_tag_idx]
        for backpointers_t in reversed(backpointers):
            best_tag_idx = backpointers_t[best_tag_idx]
            best_path.append(best_tag_idx)
        best_path.reverse()
        return score, best_path[1:]  # don't return start tag

    def tag_sentence(self, sent):
        feats = self.extract_feats(sent)
        score, best_label_sequence = self.viterbi_decode(feats)
        return score, [label_vocab.i2w[label_idx] for label_idx in best_label_sequence]

    def calc_loss(self, sent, labels):
        """ Negative log likelihood """
        feats = self.extract_feats(sent)
        fwd_score = self.forward_algorithm(feats)
        gold_score = self.score_sent_labels(feats, labels)
        return fwd_score - gold_score

    def forward_algorithm(self, feats):
        """ Computes the partition function Z with the forward algorithm """
        init_alphas = np.full((1, len(self.label_vocab)), -10000.0, dtype=float)  # init log probabilities
        init_alphas[0, self.label_vocab.sos_idx] = 0.0

        alphas = dy.parameter(self.pc_all.parameters_from_numpy(init_alphas, name='alpha'))  # so we can backprop thru
        transitions = dy.parameter(self.transitions)

        for feat in feats:  # each word is represented by a feature produced from the bilstm
            alphas_t = []  # keep track of forward variables for the timestep
            for next_tag in range(len(self.label_vocab)):
                emit_score = feat[next_tag] * dy.ones((1, len(self.label_vocab)))  # broadcast score over labels
                trans_score = dy.transpose(transitions[next_tag])
                next_score = alphas + trans_score + emit_score
                alphas_t.append(dy.logsumexp_dim(next_score, 1))
            alphas = dy.concatenate_cols(alphas_t)  # go from list of expressions to a single expression
        final_score = alphas + dy.transpose(transitions[self.label_vocab.eos_idx])
        alpha = dy.logsumexp_dim(final_score, 1)
        return alpha

    def score_sent_labels(self, feats, tags):
        """ Calculate score of sentence's labels """
        score = 0.0
        tags = [self.label_vocab.sos_idx] + tags
        transitions = dy.parameter(self.transitions)
        for i, feat in enumerate(feats):
            score += transitions[tags[i + 1]][tags[i]] + feat[tags[i + 1]]
        score += transitions[self.label_vocab.eos_idx][tags[-1]]
        return score

    def load(self, path):
        self.pc.populate(path)


def create_batches(dataset, max_batch_size):
    """
    Create batches with sentences of similar lengths to make training more efficient.
    :param dataset: dataset
    :param max_batch_size: int
    :return: batches [(start, length), ...]
    """
    dataset.sort(key=lambda t: len(t[0]), reverse=True)  # sort by sentence length (longer first)
    sentences = [x[0] for x in dataset]
    lengths = [len(x) for x in sentences]
    batches = []
    prev = lengths[0]
    prev_start = 0
    batch_size = 1
    for i in range(prev_start + 1, len(lengths)):
        if lengths[i] != prev or batch_size == max_batch_size:  # start a new batch
            batches.append((prev_start, batch_size))
            prev = lengths[i]
            prev_start = i
            batch_size = 1
        else:  # continue the batch
            batch_size += 1
    return batches


def train(model, train_data, dev_data, trainer, epochs, train_batch_size, dev_batch_size, print_every):
    # create model checkpoint directory
    model_path = "models/{}".format(datetime.datetime.now().strftime('%m-%d_%H%M'))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print("Creating training batches of size", train_batch_size)
    train_batches = create_batches(train_data, train_batch_size)
    print("Number of train batches:", len(train_batches))
    print("Creating dev batches of size", dev_batch_size)
    dev_batches = create_batches(dev_data, dev_batch_size)
    print("Number of dev batches:", len(dev_batches))
    print()
    print('Training for {} epochs on {} sentences of training data.'.format(epochs, len(train_data)))
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(epochs):
        print('EPOCH:', epoch+1)
        print("Shuffling batches")
        random.shuffle(train_batches)
        random.shuffle(dev_batches)            # TODO: delete
        train_losses, train_loss = [], 0.0
        dev_losses, dev_loss = [], 0.0
        epoch_start_time = batch_start_time = time.time()
        for batch_idx, (start, length) in enumerate(train_batches):
            dy.renew_cg()
            train_batch = train_data[start:start + length]
            for i, (sent, labels) in enumerate(train_batch):
                loss = model.calc_loss(sent, labels)
                train_losses.append(loss)  # accumulate expressions for autobatching
            # do forward pass now for autobatching
            batch_train_loss = dy.esum(train_losses)
            train_losses = []
            train_loss += batch_train_loss.value()
            batch_train_loss.backward()
            trainer.update()
            if (batch_idx + 1) % print_every == 0:
                print('--finished {} batches at {:.2f}s/sent'.format(batch_idx + 1,
                                                                    (time.time() - batch_start_time) /
                                                                     train_batch_size))
                batch_start_time = time.time()

        time_elapsed = time.time() - epoch_start_time
        print('TRAIN: loss/sent={:4f}, epoch time={:.2f}s'.format(train_loss / len(train_data), time_elapsed))

        start_time = time.time()
        for batch_idx, (start, length) in enumerate(dev_batches):
            dy.renew_cg()
            dev_batch = dev_data[start:start + length]
            for i, (sent, labels) in enumerate(dev_batch):
                loss = model.calc_loss(sent, labels)
                dev_losses.append(loss)  # accumulate expressions for autobatching
            batch_dev_loss = dy.esum(dev_losses)
            dev_losses = []
            dev_loss += batch_dev_loss.value()

        time_elapsed = time.time() - start_time
        print('DEV: loss/sent={:.4f}, time/sent={:.2f}s'.format(dev_loss / len(dev_data),
                                                                time_elapsed / len(dev_data)))
        model.pc.save("{}/iter_{}".format(model_path, epoch + 1))
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_epoch = epoch + 1
    print(best_epoch)


def decode(model, word_vocab, label_vocab, test_data, test_batch_size, output_file, as_pos_tagger):
    # TODO: autobatched decoding
    test_batches = create_batches(test_data, test_batch_size)
    labeled_sents = []
    for batch_idx, (start, length) in enumerate(test_batches):
        test_batch = test_data[start:start + length]
        if as_pos_tagger:
            # for i, (sent, pos_tags, bio) in enumerate(test_batch):
            #     dy.renew_cg()
            #     words = [word_vocab.i2w[word_idx] for word_idx in sent]
            #     gt_pos_tags = [label_vocab.i2w[label_idx] for label_idx in pos_tags]
            #     hyp_pos_tags = model.tag_sentence(sent)[1]
            #     labeled_sents.append(list(zip(words, bio, gt_pos_tags, hyp_pos_tags)))
            for i, (sent, pos_tags) in enumerate(test_batch):
                dy.renew_cg()
                words = [word_vocab.i2w[word_idx] for word_idx in sent]
                gt_pos_tags = [label_vocab.i2w[label_idx] for label_idx in pos_tags]
                hyp_pos_tags = model.tag_sentence(sent)[1]
                labeled_sents.append(list(zip(words, ['-']*len(words), ['-']*len(words), gt_pos_tags, hyp_pos_tags)))
        else:
            for i, (sent, pos_tags, chunks, labels) in enumerate(test_batch):
                dy.renew_cg()
                words = [word_vocab.i2w[word_idx] for word_idx in sent]
                gt_labels = [label_vocab.i2w[label_idx] for label_idx in labels]
                hyp_labels = model.tag_sentence(sent)[1]
                labeled_sents.append(list(zip(words, pos_tags, chunks, gt_labels, hyp_labels)))
    data.write_conll(output_file, labeled_sents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dynet flags to be processed by dynet
    parser.add_argument('--dynet-gpus')
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-autobatch')
    parser.add_argument('--dynet-seed', default=None, type=int)
    parser.add_argument('--dynet-viz')
    # data paths
    parser.add_argument('--train', default='data/train.data', type=str)
    parser.add_argument('--dev', default='data/dev.data', type=str)
    parser.add_argument('--test', default='data/test.data', type=str)
    parser.add_argument('--output', default='output/test.data.hyp')
    # model params
    parser.add_argument('--embed-dim', default=64, type=int)
    parser.add_argument('--hidden-dim', default=128, type=int, help='Dimension of the hidden state. '
                                                                    'Will be doubled (bidirectional).')
    parser.add_argument('--nlayers', default=1, type=int)
    # train params
    parser.add_argument('--train-batch-size', default=16, type=int)
    parser.add_argument('--dev-batch-size', default=8, type=int)
    parser.add_argument('--print-every', default=50, type=int, help='Print progress every N batches.')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--early-stopping', default=1, type=int, help='Whether or not to perform early stopping.')
    # loading and translating
    parser.add_argument('--from-checkpoint', type=str, default=None)
    parser.add_argument('--no-training', action='store_true', help="Don't perform training.")
    parser.add_argument('--test-batch-size', default=8, type=int)
    parser.add_argument('--as-pos-tagger', action='store_true')
    args = parser.parse_args()

    if args.dynet_seed is not None:
        random.seed(args.dynet_seed)
        np.random.seed(args.dynet_seed)

    word_vocab = Vocabulary(UNK)
    label_vocab = Vocabulary(UNK, START, STOP)
    # load a list of sentences, where each word in the list is a tuple containing the word and the bio label
    if args.as_pos_tagger:
        train_data = list(data.read_conll_spanish_sent(args.train, word_vocab, label_vocab, fields=('word', 'pos_tag')))
    else:
        train_data = list(data.read_conll_sentence(args.train, word_vocab, label_vocab, fields=('word', 'bio')))
    word_vocab.freeze()
    label_vocab.freeze()
    if args.as_pos_tagger:
        dev_data = list(data.read_conll_spanish_sent(args.dev, word_vocab, label_vocab, fields=('word', 'pos_tag')))
    else:
        dev_data = list(data.read_conll_sentence(args.dev, word_vocab, label_vocab, fields=('word', 'bio')))

    model = BiLSTMCRF(args.embed_dim, args.hidden_dim, args.nlayers, word_vocab, label_vocab)
    if args.from_checkpoint:
        model.load(args.from_checkpoint)

    if not args.no_training:
        trainer = dy.AdamTrainer(model.pc_all)
        train(model, train_data, dev_data, trainer, args.epochs, args.train_batch_size, args.dev_batch_size,
              args.print_every)

    # output results to file
    if args.as_pos_tagger:
        test_data = list(data.read_conll_spanish_sent(args.test, word_vocab, label_vocab,
                                                      fields=('word', 'pos_tag')))
    else:
        test_data = list(data.read_conll_sentence(args.test, word_vocab, label_vocab,
                                                  fields=('word', 'pos_tag', 'chunk', 'bio')))
    decode(model, word_vocab, label_vocab, test_data, args.test_batch_size, args.output, args.as_pos_tagger)
