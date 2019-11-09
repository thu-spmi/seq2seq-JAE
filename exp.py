# coding=utf-8

from __future__ import print_function

import argparse

import traceback

import numpy as np
import time
import math
import os
import sys
import pickle as pkl

import torch

from torch.autograd import Variable

import evaluation
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from components.dataset import Dataset, Example
import model.nn_utils as nn_utils
from model.neural_lm import LSTMLanguageModel

from model.parser import Parser
from model.prior import UniformPrior, LSTMPrior
from model.reconstruction_model import Reconstructor
from model.struct_jae import JAE,JAE_cache
from model.struct_vae import StructVAE, StructVAE_LMBaseline, StructVAE_SrcLmAndLinearBaseline


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='use gpu')
    parser.add_argument('--lang', choices=['python', 'lambda_dcs'], default='python')
    parser.add_argument('--mode', choices=['train', 'train_decoder', 'train_semi_jae', 'train_semi_vae', 'test'], default='train', help='run mode')

    parser.add_argument('--lstm', choices=['lstm', 'lstm_with_dropout'], default='lstm')

    parser.add_argument('--load_model', default=None, type=str, help='load a pre-trained model')

    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--unsup_batch_size', default=10, type=int)
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--sample_size', default=5, type=int, help='sample size')
    parser.add_argument('--embed_size', default=128, type=int, help='size of word embeddings')
    parser.add_argument('--action_embed_size', default=128, type=int, help='size of word embeddings')
    parser.add_argument('--field_embed_size', default=64, type=int, help='size of word embeddings')
    parser.add_argument('--type_embed_size', default=64, type=int, help='size of word embeddings')
    parser.add_argument('--ptrnet_hidden_dim', default=32, type=int)
    parser.add_argument('--hidden_size', default=256, type=int, help='size of LSTM hidden states')
    parser.add_argument('--dropout', default=0., type=float, help='dropout rate')
    parser.add_argument('--decoder_word_dropout', default=0.3, type=float, help='word dropout on decoder')
    parser.add_argument('--kl_anneal', default=False, action='store_true')
    parser.add_argument('--alpha', default=0.1, type=float)

    parser.add_argument('--asdl_file', type=str)
    parser.add_argument('--vocab', type=str, help='path of the serialized vocabulary')
    parser.add_argument('--train_src', type=str, help='path to the training source file')
    parser.add_argument('--unlabeled_file', type=str, help='path to the training source file')
    parser.add_argument('--unlabeled_file2', type=str, help='path to the training source file')
    parser.add_argument('--train_file', type=str, help='path to the training target file')
    parser.add_argument('--dev_file', type=str, help='path to the dev source file')
    parser.add_argument('--test_file', type=str, help='path to the test target file')
    parser.add_argument('--prior_lm_path', type=str, help='path to the prior LM')

    # self-training
    parser.add_argument('--load_decode_results', default=None, type=str)

    # semi-supervised learning arguments
    parser.add_argument('--load_decoder', default=None, type=str)
    parser.add_argument('--load_src_lm', default=None, type=str)
    parser.add_argument('--load_prior', type=str, default=None)
    parser.add_argument('--clip_learning_signal', type=float, default=None)
    parser.add_argument('--begin_semisup_after_dev_acc', type=float, default=0., help='begin semi-supervised learning after'
                                                                                    'we have reached certain dev performance')

    parser.add_argument('--decode_max_time_step', default=100, type=int, help='maximum number of time steps used '
                                                                              'in decoding and sampling')
    parser.add_argument('--unsup_loss_weight', default=1., type=float, help='loss of unsupervised learning weight')

    parser.add_argument('--valid_metric', default='sp_acc', choices=['nlg_bleu', 'sp_acc'],
                        help='metric used for validation')
    parser.add_argument('--log_every', default=10, type=int, help='every n iterations to log training statistics')

    parser.add_argument('--save_to', default='model', type=str, help='save trained model to')
    parser.add_argument('--save_decode_to', default=None, type=str, help='save decoding results to file')
    parser.add_argument('--patience', default=5, type=int, help='training patience')
    parser.add_argument('--max_num_trial', default=10, type=int)
    parser.add_argument('--uniform_init', default=None, type=float,
                        help='if specified, use uniform initialization for all parameters')
    parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    parser.add_argument('--max_epoch', default=-1, type=int, help='maximum number of training epoches')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float,
                        help='decay learning rate if the validation performance drops')
    parser.add_argument('--lr_decay_after_epoch', default=5, type=int)
    parser.add_argument('--reset_optimizer', action='store_true', default=False)
    parser.add_argument('--cache', default=0, type=int) #whether to cache samples for JAE
    parser.add_argument('--bi_direction', default=0, type=int) #whether to use bi-JAE
    parser.add_argument('--moves', default=5, type=int) #multiple moves of MIS for JAE
    parser.add_argument('--train_opt', default="reinforce", type=str, choices=['reinforce', 'st_gumbel'])

    args = parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed) 
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed * 13 // 7)

    return args


def train(args):
    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    transition_system = TransitionSystem.get_class_by_lang(args.lang)(grammar)
    train_set = Dataset.from_bin_file(args.train_file)
    dev_set = Dataset.from_bin_file(args.dev_file)
    vocab = pkl.load(open(args.vocab,'rb'))

    model = Parser(args, vocab, transition_system)
    model.train()
    if args.cuda: model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = 0.
    history_dev_scores = []
    num_trial = patience = 0
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= args.decode_max_time_step]

            train_iter += 1
            optimizer.zero_grad()

            loss = -model.score(batch_examples)
            # print(loss.data)
            loss_val = torch.sum(loss).data[0]
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            loss.backward()

            # clip gradient
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)

            optimizer.step()

            if train_iter % args.log_every == 0:
                print('[Iter %d] encoder loss=%.5f' %
                      (train_iter,
                       report_loss / report_examples),
                      file=sys.stderr)

                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)
        # model_file = args.save_to + '.iter%d.bin' % train_iter
        # print('save model to [%s]' % model_file, file=sys.stderr)
        # model.save(model_file)

        # perform validation
        print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
        eval_start = time.time()
        eval_results = evaluation.evaluate(dev_set.examples, model, args, verbose=True)
        dev_acc = eval_results['accuracy']
        print('[Epoch %d] code generation accuracy=%.5f took %ds' % (epoch, dev_acc, time.time() - eval_start), file=sys.stderr)
        is_better = history_dev_scores == [] or dev_acc > max(history_dev_scores)
        history_dev_scores.append(dev_acc)

        if is_better:
            patience = 0
            model_file = args.save_to + '.bin'
            print('save currently the best model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')
        elif epoch == args.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            exit(0)
        elif patience < args.patience:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if patience == args.patience:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args.max_num_trial:
                print('early stop!', file=sys.stderr)
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(args.save_to + '.bin', map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if args.cuda: model = model.cuda()

            # load optimizers
            if args.reset_optimizer:
                print('reset optimizer', file=sys.stderr)
                optimizer = torch.optim.Adam(model.inference_model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0


def train_decoder(args):
    train_set = Dataset.from_bin_file(args.train_file)
    dev_set = Dataset.from_bin_file(args.dev_file)
    vocab = pkl.load(open(args.vocab,'rb'))

    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    transition_system = TransitionSystem.get_class_by_lang(args.lang)(grammar)

    model = Reconstructor(args, vocab, transition_system)
    model.train()
    if args.cuda: model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def evaluate_ppl():
        model.eval()
        cum_loss = 0.
        cum_tgt_words = 0.
        for batch in dev_set.batch_iter(args.batch_size):
            loss = -model.score(batch).sum()
            cum_loss += loss.data[0]
            cum_tgt_words += sum(len(e.src_sent) + 1 for e in batch)  # add ending </s>

        ppl = np.exp(cum_loss / cum_tgt_words)
        model.train()
        return ppl

    print('begin training decoder, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = 0.
    history_dev_scores = []
    num_trial = patience = 0
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= args.decode_max_time_step]
            # batch_examples = [e for e in train_set.examples if e.idx in [10192, 10894, 9706, 4659, 5609, 1442, 5849, 10644, 4592, 1875]]

            train_iter += 1
            optimizer.zero_grad()

            loss = -model.score(batch_examples)
            # print(loss.data)
            loss_val = torch.sum(loss).data[0]
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)

            optimizer.step()

            if train_iter % args.log_every == 0:
                print('[Iter %d] encoder loss=%.5f' %
                      (train_iter,
                       report_loss / report_examples),
                      file=sys.stderr)

                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)
        # model_file = args.save_to + '.iter%d.bin' % train_iter
        # print('save model to [%s]' % model_file, file=sys.stderr)
        # model.save(model_file)

        # perform validation
        print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
        eval_start = time.time()
        # evaluate ppl
        ppl = evaluate_ppl()
        print('[Epoch %d] ppl=%.5f took %ds' % (epoch, ppl, time.time() - eval_start), file=sys.stderr)
        dev_acc = -ppl
        is_better = history_dev_scores == [] or dev_acc > max(history_dev_scores)
        history_dev_scores.append(dev_acc)

        if is_better:
            patience = 0
            model_file = args.save_to + '.bin'
            print('save currently the best model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')
        elif patience < args.patience:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if patience == args.patience:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args.max_num_trial:
                print('early stop!', file=sys.stderr)
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(args.save_to + '.bin', map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if args.cuda: model = model.cuda()

            # load optimizers
            if args.reset_optimizer:
                print('reset optimizer', file=sys.stderr)
                optimizer = torch.optim.Adam(model.inference_model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0


def train_semi_jae(args):

    bi_direction=args.bi_direction

    encoder_params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    decoder_params = torch.load(args.load_decoder, map_location=lambda storage, loc: storage)

    print('loaded encoder at %s' % args.load_model, file=sys.stderr)
    print('loaded decoder at %s' % args.load_decoder, file=sys.stderr)

    transition_system = encoder_params['transition_system']
    encoder_params['args'].cuda = decoder_params['args'].cuda = args.cuda

    encoder = Parser(encoder_params['args'], encoder_params['vocab'], transition_system)
    encoder.load_state_dict(encoder_params['state_dict'])
    decoder = Reconstructor(decoder_params['args'], decoder_params['vocab'], transition_system)
    decoder.load_state_dict(decoder_params['state_dict'])

    zprior = LSTMPrior.load(args.load_prior, transition_system=transition_system, cuda=args.cuda)
    print('loaded p(z) prior at %s' % args.load_prior, file=sys.stderr)
    # freeze prior parameters
    for p in zprior.parameters():
        p.requires_grad = False
    zprior.eval()
    xprior=LSTMLanguageModel.load(args.load_src_lm)
    print('loaded p(x) prior at %s' % args.load_src_lm, file=sys.stderr)
    xprior.eval()

    if args.cache:
        jae=JAE_cache(encoder, decoder, zprior,xprior, args)
    else:
        jae = JAE(encoder, decoder, zprior,xprior, args)

    jae.train()
    encoder.train()
    decoder.train()
    if args.cuda: jae.cuda()

    labeled_data = Dataset.from_bin_file(args.train_file)
    # labeled_data.examples = labeled_data.examples[:10]
    unlabeled_data = Dataset.from_bin_file(args.unlabeled_file)   # pretend they are un-labeled!
    dev_set = Dataset.from_bin_file(args.dev_file)
    # dev_set.examples = dev_set.examples[:10]

    optimizer = torch.optim.Adam([p for p in jae.parameters() if p.requires_grad], lr=args.lr)

    print('*** begin semi-supervised training %d labeled examples, %d unlabeled examples ***' %
          (len(labeled_data), len(unlabeled_data)), file=sys.stderr)
    report_encoder_loss = report_decoder_loss = report_examples = 0.
    report_unsup_examples = report_unsup_encoder_loss = report_unsup_decoder_loss = report_unsup_baseline_loss = 0.
    patience = 0
    num_trial = 1
    epoch = train_iter = 0
    history_dev_scores = []
    while True:
        epoch += 1
        epoch_begin = time.time()
        unlabeled_examples_iter = unlabeled_data.batch_iter(batch_size=args.unsup_batch_size, shuffle=True)
        for labeled_examples in labeled_data.batch_iter(batch_size=args.batch_size, shuffle=True):
            labeled_examples = [e for e in labeled_examples if len(e.tgt_actions) <= args.decode_max_time_step]

            train_iter += 1

            optimizer.zero_grad()

            report_examples += len(labeled_examples)

            sup_encoder_loss = -encoder.score(labeled_examples)
            sup_decoder_loss = -decoder.score(labeled_examples)

            report_encoder_loss += sup_encoder_loss.sum().data[0]
            report_decoder_loss += sup_decoder_loss.sum().data[0]

            sup_encoder_loss = torch.mean(sup_encoder_loss)
            sup_decoder_loss = torch.mean(sup_decoder_loss)

            sup_loss = sup_encoder_loss + sup_decoder_loss

            # compute unsupervised loss

            try:
                unlabeled_examples = next(unlabeled_examples_iter)
            except StopIteration:
                # if finished unlabeled data stream, restart it
                unlabeled_examples_iter = unlabeled_data.batch_iter(batch_size=args.batch_size, shuffle=True)
                unlabeled_examples = next(unlabeled_examples_iter)
                unlabeled_examples = [e for e in unlabeled_examples if len(e.tgt_actions) <= args.decode_max_time_step]


            unsup_encoder_loss, unsup_decoder_loss, meta_data = jae.get_unsupervised_loss(
                unlabeled_examples,args.moves)
            if bi_direction:
                unsup_encoder_loss_back, unsup_decoder_loss_back, meta_data_back = jae.get_unsupervised_loss_backward(
                    unlabeled_examples, args.moves)

            nan = False
            if nn_utils.isnan(sup_loss.data):
                print('Nan in sup_loss')
                nan = True
            if nn_utils.isnan(unsup_encoder_loss.data):
                print('Nan in unsup_encoder_loss!', file=sys.stderr)
                nan = True
            if nn_utils.isnan(unsup_decoder_loss.data):
                print('Nan in unsup_decoder_loss!', file=sys.stderr)
                nan = True
            if bi_direction:
                if nn_utils.isnan(unsup_encoder_loss_back.data):
                    print('Nan in unsup_encoder_loss_back!', file=sys.stderr)
                    nan = True
                if nn_utils.isnan(unsup_decoder_loss_back.data):
                    print('Nan in unsup_decoder_loss_back!', file=sys.stderr)
                    nan = True

            if nan:
                continue
            if bi_direction:
                report_unsup_encoder_loss += (unsup_encoder_loss.sum().data[0] + unsup_encoder_loss_back.sum().data[0])
                report_unsup_decoder_loss += (unsup_decoder_loss.sum().data[0] + unsup_decoder_loss_back.sum().data[0])
            else:
                report_unsup_encoder_loss += unsup_encoder_loss.sum().data[0]
                report_unsup_decoder_loss += unsup_decoder_loss.sum().data[0]
            report_unsup_examples += unsup_encoder_loss.size(0)

            if bi_direction:
                unsup_loss = torch.mean(unsup_encoder_loss) + torch.mean(unsup_decoder_loss) + torch.mean(unsup_encoder_loss_back) + torch.mean(unsup_decoder_loss_back)
            else:
                unsup_loss = torch.mean(unsup_encoder_loss) + torch.mean(unsup_decoder_loss)
            loss = sup_loss + args.unsup_loss_weight * unsup_loss

            loss.backward()
            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(jae.parameters(), args.clip_grad)
            optimizer.step()
            if train_iter % args.log_every == 0:
                print('[Iter %d] supervised: encoder loss=%.5f, decoder loss=%.5f' %
                      (train_iter,
                       report_encoder_loss / report_examples,
                       report_decoder_loss / report_examples),
                      file=sys.stderr)

                print('[Iter %d] unsupervised: encoder loss=%.5f, decoder loss=%.5f, baseline loss=%.5f' %
                      (train_iter,
                       report_unsup_encoder_loss / report_unsup_examples,
                       report_unsup_decoder_loss / report_unsup_examples,
                       report_unsup_baseline_loss / report_unsup_examples),
                      file=sys.stderr)

                samples = meta_data['samples']
                for v in meta_data.values():
                    if isinstance(v, Variable): v.cpu()
                for i, sample in enumerate(samples[:1]):
                    print('\t[%s] Source: %s' % (sample.idx, ' '.join(sample.src_sent)), file=sys.stderr)
                    print('\t[%s] Code: \n%s' % (sample.idx, sample.tgt_code), file=sys.stderr)
                    ref_example = [e for e in unlabeled_examples if e.idx == int(sample.idx[:sample.idx.index('-')])][0]
                    print('\t[%s] Gold Code: \n%s' % (sample.idx, ref_example.tgt_code), file=sys.stderr)
                    print('\t[%s] Log p(z|x): %f' % (sample.idx, meta_data['encoding_scores'][i].data[0]), file=sys.stderr)
                    print('\t[%s] Log p(x|z): %f' % (sample.idx, meta_data['reconstruction_scores'][i].data[0]), file=sys.stderr)
                    print('\t[%s] Encoder Loss: %f' % (sample.idx, unsup_encoder_loss[i].data[0]), file=sys.stderr)
                    print('\t**************************', file=sys.stderr)

                report_encoder_loss = report_decoder_loss = report_examples = 0.
                report_unsup_encoder_loss = report_unsup_decoder_loss = report_unsup_baseline_loss = report_unsup_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)
        # perform validation
        print('[Epoch %d] begin validation' % epoch, file=sys.stderr)

        eval_start = time.time()
        eval_results = evaluation.evaluate(dev_set.examples, encoder, args, verbose=True)
        encoder.train()
        dev_acc = eval_results['accuracy']
        print('[Epoch %d] code generation accuracy=%.5f took %ds' % (epoch, dev_acc, time.time() - eval_start),
              file=sys.stderr)
        is_better = history_dev_scores == [] or dev_acc > max(history_dev_scores)
        history_dev_scores.append(dev_acc)

        if is_better:
            patience = 0
            model_file = args.save_to + '.bin'
            print('save currently the best model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            jae.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')
        elif epoch == args.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            exit(0)
        elif patience < args.patience:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if patience == args.patience:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args.max_num_trial:
                print('early stop!', file=sys.stderr)
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load best model's parameters
            jae.load_parameters(args.save_to + '.bin')
            if args.cuda: jae = jae.cuda()

            # load optimizers
            if args.reset_optimizer:
                print('reset to a new infer_optimizer', file=sys.stderr)
                optimizer = torch.optim.Adam([p for p in jae.parameters() if p.requires_grad], lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0

def train_semi_vae(args):
    encoder_params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    decoder_params = torch.load(args.load_decoder, map_location=lambda storage, loc: storage)

    print('loaded encoder at %s' % args.load_model, file=sys.stderr)
    print('loaded decoder at %s' % args.load_decoder, file=sys.stderr)

    transition_system = encoder_params['transition_system']
    encoder_params['args'].cuda = decoder_params['args'].cuda = args.cuda

    encoder = Parser(encoder_params['args'], encoder_params['vocab'], transition_system)
    encoder.load_state_dict(encoder_params['state_dict'])
    decoder = Reconstructor(decoder_params['args'], decoder_params['vocab'], transition_system)
    decoder.load_state_dict(decoder_params['state_dict'])


    prior = LSTMPrior.load(args.load_prior, transition_system=transition_system, cuda=args.cuda)
    print('loaded prior at %s' % args.load_prior, file=sys.stderr)
    # freeze prior parameters
    for p in prior.parameters():
        p.requires_grad = False
    prior.eval()


    src_lm = LSTMLanguageModel.load(args.load_src_lm)
    print('loaded source LM at %s' % args.load_src_lm, file=sys.stderr)
    structVAE = StructVAE_LMBaseline(encoder, decoder, prior, src_lm, args)

    structVAE.train()
    if args.cuda: structVAE.cuda()

    labeled_data = Dataset.from_bin_file(args.train_file)
    unlabeled_data = Dataset.from_bin_file(args.unlabeled_file)   # pretend they are un-labeled!
    dev_set = Dataset.from_bin_file(args.dev_file)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, structVAE.parameters()), lr=args.lr)

    print('*** begin semi-supervised training %d labeled examples, %d unlabeled examples ***' %
          (len(labeled_data), len(unlabeled_data)), file=sys.stderr)
    report_encoder_loss = report_decoder_loss = report_examples = 0.
    report_unsup_examples = report_unsup_encoder_loss = report_unsup_decoder_loss = report_unsup_baseline_loss = 0.
    patience = 0
    num_trial = 1
    epoch = train_iter = 0
    history_dev_scores = []
    while True:
        epoch += 1
        epoch_begin = time.time()
        unlabeled_examples_iter = unlabeled_data.batch_iter(batch_size=args.unsup_batch_size, shuffle=True)

        for labeled_examples in labeled_data.batch_iter(batch_size=args.batch_size, shuffle=True):
            labeled_examples = [e for e in labeled_examples if len(e.tgt_actions) <= args.decode_max_time_step]

            train_iter += 1
            optimizer.zero_grad()
            report_examples += len(labeled_examples)

            sup_encoder_loss = -encoder.score(labeled_examples)
            sup_decoder_loss = -decoder.score(labeled_examples)

            report_encoder_loss += sup_encoder_loss.sum().data[0]
            report_decoder_loss += sup_decoder_loss.sum().data[0]

            sup_encoder_loss = torch.mean(sup_encoder_loss)
            sup_decoder_loss = torch.mean(sup_decoder_loss)

            sup_loss = sup_encoder_loss + sup_decoder_loss

            # compute unsupervised loss
            try:
                unlabeled_examples = next(unlabeled_examples_iter)
            except StopIteration:
                # if finished unlabeled data stream, restart it
                unlabeled_examples_iter = unlabeled_data.batch_iter(batch_size=args.batch_size, shuffle=True)
                unlabeled_examples = next(unlabeled_examples_iter)
                unlabeled_examples = [e for e in unlabeled_examples if len(e.tgt_actions) <= args.decode_max_time_step]

            try:
                unsup_encoder_loss, unsup_decoder_loss, unsup_baseline_loss, meta_data = structVAE.get_unsupervised_loss(
                    unlabeled_examples)

                nan = False
                if nn_utils.isnan(sup_loss.data):
                    print('Nan in sup_loss')
                    nan = True
                if nn_utils.isnan(unsup_encoder_loss.data):
                    print('Nan in unsup_encoder_loss!', file=sys.stderr)
                    nan = True
                if nn_utils.isnan(unsup_decoder_loss.data):
                    print('Nan in unsup_decoder_loss!', file=sys.stderr)
                    nan = True
                if nn_utils.isnan(unsup_baseline_loss.data):
                    print('Nan in unsup_baseline_loss!', file=sys.stderr)
                    nan = True

                if nan:
                    # torch.save((unsup_encoder_loss, unsup_decoder_loss, unsup_baseline_loss, meta_data), 'nan_data.bin')
                    continue

                report_unsup_encoder_loss += unsup_encoder_loss.sum().data[0]
                report_unsup_decoder_loss += unsup_decoder_loss.sum().data[0]
                report_unsup_baseline_loss += unsup_baseline_loss.sum().data[0]
                report_unsup_examples += unsup_encoder_loss.size(0)
            except ValueError as e:
                print(e.message, file=sys.stderr)
                continue

            unsup_loss = torch.mean(unsup_encoder_loss) + torch.mean(unsup_decoder_loss) + torch.mean(unsup_baseline_loss)

            loss = sup_loss + args.unsup_loss_weight * unsup_loss

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(structVAE.parameters(), args.clip_grad)
            optimizer.step()

            if train_iter % args.log_every == 0:
                print('[Iter %d] supervised: encoder loss=%.5f, decoder loss=%.5f' %
                      (train_iter,
                       report_encoder_loss / report_examples,
                       report_decoder_loss / report_examples),
                      file=sys.stderr)

                print('[Iter %d] unsupervised: encoder loss=%.5f, decoder loss=%.5f, baseline loss=%.5f' %
                      (train_iter,
                       report_unsup_encoder_loss / report_unsup_examples,
                       report_unsup_decoder_loss / report_unsup_examples,
                       report_unsup_baseline_loss / report_unsup_examples),
                      file=sys.stderr)

                if isinstance(structVAE, StructVAE_LMBaseline):
                    print('[Iter %d] baseline: source LM b_lm_weight: %.3f, b: %.3f' % (train_iter,
                                                                                        structVAE.b_lm_weight.data[0],
                                                                                        structVAE.b.data[0]),
                          file=sys.stderr)

                samples = meta_data['samples']
                for v in meta_data.itervalues():
                    if isinstance(v, Variable): v.cpu()
                for i, sample in enumerate(samples[:1]):
                    print('\t[%s] Source: %s' % (sample.idx, ' '.join(sample.src_sent)), file=sys.stderr)
                    print('\t[%s] Code: \n%s' % (sample.idx, sample.tgt_code), file=sys.stderr)
                    ref_example = [e for e in unlabeled_examples if e.idx == int(sample.idx[:sample.idx.index('-')])][0]
                    print('\t[%s] Gold Code: \n%s' % (sample.idx, ref_example.tgt_code), file=sys.stderr)
                    print('\t[%s] Log p(z|x): %f' % (sample.idx, meta_data['encoding_scores'][i].data[0]), file=sys.stderr)
                    print('\t[%s] Log p(x|z): %f' % (sample.idx, meta_data['reconstruction_scores'][i].data[0]), file=sys.stderr)
                    print('\t[%s] KL term: %f' % (sample.idx, meta_data['kl_term'][i].data[0]), file=sys.stderr)
                    print('\t[%s] Prior: %f' % (sample.idx, meta_data['prior'][i].data[0]), file=sys.stderr)
                    print('\t[%s] baseline: %f' % (sample.idx, meta_data['baseline'][i].data[0]), file=sys.stderr)
                    print('\t[%s] Raw Learning Signal: %f' % (sample.idx, meta_data['raw_learning_signal'][i].data[0]), file=sys.stderr)
                    print('\t[%s] Learning Signal - baseline: %f' % (sample.idx, meta_data['learning_signal'][i].data[0]), file=sys.stderr)
                    print('\t[%s] Encoder Loss: %f' % (sample.idx, unsup_encoder_loss[i].data[0]), file=sys.stderr)
                    print('\t**************************', file=sys.stderr)

                report_encoder_loss = report_decoder_loss = report_examples = 0.
                report_unsup_encoder_loss = report_unsup_decoder_loss = report_unsup_baseline_loss = report_unsup_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)
        # perform validation
        print('[Epoch %d] begin validation' % epoch, file=sys.stderr)

        eval_start = time.time()
        eval_results = evaluation.evaluate(dev_set.examples, encoder, args, verbose=True)
        dev_acc = eval_results['accuracy']
        print('[Epoch %d] code generation accuracy=%.5f took %ds' % (epoch, dev_acc, time.time() - eval_start),
              file=sys.stderr)
        is_better = history_dev_scores == [] or dev_acc > max(history_dev_scores)
        history_dev_scores.append(dev_acc)

        if is_better:
            patience = 0
            model_file = args.save_to + '.bin'
            print('save currently the best model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            structVAE.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')
        elif epoch == args.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            exit(0)
        elif patience < args.patience:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if patience == args.patience:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args.max_num_trial:
                print('early stop!', file=sys.stderr)
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load best model's parameters
            structVAE.load_parameters(args.save_to + '.bin')
            if args.cuda: structVAE = structVAE.cuda()

            # load optimizers
            if args.reset_optimizer:
                print('reset to a new infer_optimizer', file=sys.stderr)
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, structVAE.parameters()), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0


def test(args):
    test_set = Dataset.from_bin_file(args.test_file)
    assert args.load_model

    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    vocab = params['vocab']
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_state = params['state_dict']
    saved_args.cuda = args.cuda

    parser = Parser(saved_args, vocab, transition_system)
    parser.load_state_dict(saved_state)

    if args.cuda: parser = parser.cuda()
    parser.eval()

    eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, args,
                                                       verbose=True, return_decode_result=True)
    print(eval_results, file=sys.stderr)
    if args.save_decode_to:
        pkl.dump(decode_results, open(args.save_decode_to, 'wb'))


if __name__ == '__main__':
    args = init_config()

    print(args, file=sys.stderr)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'train_decoder':
        train_decoder(args)
    elif args.mode == 'train_semi_vae':
        train_semi_vae(args)
    elif args.mode == 'train_semi_jae':
        train_semi_jae(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise RuntimeError('unknown mode')
