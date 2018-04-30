import time
import argparse
import random
import numpy as np
random.seed(666)
np.random.seed(666)
import dynet as dy
from vocab import Vocabulary
import data
import datetime
import os


class BiLSTMCRF:
    def __init__(self, embed_dim, hidden_dim, nlayers, word_vocab, label_vocab):
        self.pc = dy.ParameterCollection()
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.embeddings = self.pc.add_lookup_parameters((len(word_vocab), embed_dim))
        self.bilstm = dy.BiRNNBuilder(nlayers, embed_dim, hidden_dim * 2, self.pc, dy.LSTMBuilder)
        self.W_output = self.pc.add_parameters((len(label_vocab), hidden_dim * 2))
        self.b_output = self.pc.add_parameters(len(label_vocab))
        self.transitions = self.pc.add_parameters((len(label_vocab), len(label_vocab)))

    def calc_label_scores(self, sent):
        dy.renew_cg()
        bilstm_outputs = self.bilstm.transduce([self.embeddings[word_idx] for word_idx in sent])
        label_scores = dy.affine_transform([dy.parameter(self.b_output), dy.parameter(self.W_output),
                                            dy.concatenate_cols(bilstm_outputs)])
        return dy.transpose(label_scores)

    def viterbi_decode(self, emission_prob):
        pass

    def tag_sentence(self, sent):
        label_scores = self.calc_label_scores(sent)
        score, best_label_sequence = self.viterbi_decode(label_scores)

    def calc_loss(self, sent, labels):
        # calc neg log likelihood
        label_scores = self.calc_label_scores(sent)
        forward_score = self.forward_alg(label_scores)
        gold_score = self.score_sent_labels(label_scores, labels)
        return forward_score - gold_score

    def forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = np.full((1, len(self.label_vocab)), -10000.0, dtype=float)  # init log probabilities
        # START_TAG has all of the score.
        init_alphas[0, self.label_vocab.sos_idx] = 0.0

        # Wrap in a variable so that we will get automatic backprop
        fwd_param = dy.parameter(self.pc.parameters_from_numpy(init_alphas))

        transitions = dy.parameter(self.transitions)

        # Iterate through the sentence       # TODO: step through to figure out reshaping dims
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(len(self.label_vocab)):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                # emit_score = feat[next_tag].view(1, -1).expand(1, len(self.label_vocab))
                emit_score = feat[next_tag] * dy.ones((1, len(self.label_vocab)))
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                # trans_score = self.transitions[next_tag].view(1, -1)
                trans_score = dy.transpose(transitions[next_tag])
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = fwd_param + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                # alphas_t.append(log_sum_exp(next_tag_var).view(1))
                alphas_t.append(dy.logsumexp([e for e in next_tag_var[0]]))
            # fwd_param = torch.cat(alphas_t).view(1, -1)
            fwd_param = dy.concatenate(alphas_t)  # go from list of expressions to an expression
        terminal_var = fwd_param + transitions[self.label_vocab.eos_idx]
        alpha = dy.logsumexp([e for e in terminal_var])
        return alpha

    def score_sent_labels(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = 0.0
        tags = [self.label_vocab.sos_idx] + tags
        transitions = dy.parameter(self.transitions)
        for i, feat in enumerate(feats):
            # print(tags[i+1], tags[i])
            t = transitions[tags[i + 1]][tags[i]]
            f = feat[tags[i + 1]]
            score += t + f
        score += transitions[self.label_vocab.eos_idx][tags[-1]]
        return score


def train(model, train_data, dev_data, trainer, epochs, ):
    avg_loss_time = avg_back_time = avg_update_time = 0
    print('Training for {} epochs on {} sentences of training data.'.format(epochs, len(train_data)))
    for epoch in range(epochs):
        print('EPOCH:', epoch+1)
        train_loss = dev_loss = 0
        start_time = time.time()
        for i, (sent, labels) in enumerate(train_data):
            loss = model.calc_loss(sent, labels)
            train_loss += loss.value()
            loss.backward()
            trainer.update()
            if i+1 % 1000 == 0:
                print("--finished {} iters".format(i))
        time_elapsed = time.time() - start_time
        print('TRAIN: loss={:4f}, time={:.2f}s'.format(train_loss, time_elapsed))

        start_time = time.time()
        for sent, labels in dev_data:
            dev_loss += model.calc_loss(sent, labels).value()
        time_elapsed = time.time() - start_time
        print('DEV: loss={:.4f}, time={:.2f}s'.format(dev_loss, time_elapsed))
        # save model checkpoint
        model_path = "models/{}".format(datetime.datetime.now().strftime('%m-%d_%H%M'))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.pc.save("{}/iter_{}".format(model_path, epoch + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dynet flags to be processed by dynet
    parser.add_argument('--dynet-gpus')
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-autobatch')
    parser.add_argument('--dynet-seed')
    # data paths
    parser.add_argument('--train', default='data/train.data', type=str)
    parser.add_argument('--dev', default='data/dev.data', type=str)
    # model params
    parser.add_argument('--embed-dim', default=300, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int, help='Dimension of the hidden state. '
                                                                    'Will be doubled (bidirectional).')
    parser.add_argument('--nlayers', default=1, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    # train params
    parser.add_argument('--epochs', default=5, type=int)
    args = parser.parse_args()

    word_vocab = Vocabulary('<UNK>')
    label_vocab = Vocabulary('<UNK>', '<START>', '<STOP>')
    train_data = list(data.read_conll_sentence(args.train, word_vocab, label_vocab))  # list of (sentence, labels)
    word_vocab.freeze()
    label_vocab.freeze()
    dev_data = list(data.read_conll_sentence(args.dev, word_vocab, label_vocab))

    model = BiLSTMCRF(args.embed_dim, args.hidden_dim, args.nlayers, word_vocab, label_vocab)

    trainer = dy.AdamTrainer(model.pc)
    trainer.set_clip_threshold(5)
    train(model, train_data, dev_data, trainer, args.epochs)
