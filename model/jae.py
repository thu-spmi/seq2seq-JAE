# coding=utf-8


import sys
import traceback

import os
import copy
import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np

from components.dataset import Example
from model.prior import UniformPrior
from parser import *
from reconstruction_model import *



class JAE(nn.Module): #JAE without caching samples
    def __init__(self, encoder, decoder, zprior,xprior, args):
        super(JAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.zprior = zprior
        self.xprior=xprior
        self.transition_system = self.encoder.transition_system
        self.args = args

    def get_unsupervised_loss_forward(self, examples,moves):
        #compute loss for x->z->x path

        MIS_samples,sample_scores,reconstruction_scores=self.MIS(examples,moves)
        encoder_loss = -sample_scores
        decoder_loss = -reconstruction_scores

        meta_data = {'samples': MIS_samples,
                     'reconstruction_scores': reconstruction_scores,
                     'encoding_scores': sample_scores}
        return encoder_loss, decoder_loss, meta_data

    def MIS(self,examples,moves):

        # apply MIS on proposals of q_{\phi}(z|x)
        samples, sample_scores_with_grad= self.infer(examples,moves)
        reconstruction_scores_with_grad=self.decoder.score(samples)
        reconstruction_scores = reconstruction_scores_with_grad.data.cpu().numpy()
        sample_scores=sample_scores_with_grad.data.cpu().numpy()

        prior_scores = self.zprior([e.tgt_code for e in samples]).data.cpu().numpy()
        w = (reconstruction_scores + prior_scores - sample_scores).reshape(-1, moves)

        ind=[]

        cur=w[:,0]
        ind_temp=np.arange(0,len(samples),moves)
        for i in range(1,moves):
            u=np.random.uniform(0,1,len(w))
            accept = u < np.exp(w[:, i] - cur)
            cur=cur*(1-accept)+w[:,i]*accept
            ind_temp=ind_temp*(1-accept)+np.arange(i,len(samples),moves)*accept
            ind+=[int(i) for i in ind_temp]

        MIS_samples = [samples[i] for i in ind]
        ind = Variable(torch.LongTensor(ind), volatile=True).cuda()
        sample_scores_with_grad = torch.gather(sample_scores_with_grad, dim=0, index=ind)
        reconstruction_scores_with_grad = torch.gather(reconstruction_scores_with_grad, dim=0, index=ind)
        return MIS_samples,sample_scores_with_grad,reconstruction_scores_with_grad


    def infer(self, examples,sample_size):
        # get samples of q_{\phi}(z|x) on evaluating mode
        was_training = self.encoder.training
        self.encoder.eval()
        hypotheses = [self.encoder.parse_sample(e.src_sent) for e in examples]

        if len(hypotheses) == 0:
            raise ValueError('No candidate hypotheses.')

        if was_training: self.encoder.train()

        # some source may not have corresponding samples, so we only retain those that have sampled logical forms
        sampled_examples = []
        for e_id, (example, hyps) in enumerate(zip(examples, hypotheses)):
            if len(hyps)!=sample_size:
                print('infer not valid sample size expected %d but %d'%(sample_size,len(hyps)))
            samples_temp = []
            for hyp_id, hyp in enumerate(hyps):
                try:
                    code = self.transition_system.ast_to_surface_code(hyp.tree)
                    self.transition_system.tokenize_code(code)  # make sure the code is tokenizable!
                    if len(code)==0:
                        continue
                    sampled_example = Example(idx='%d-sample%d' % (example.idx, hyp_id),
                                              src_sent=example.src_sent,
                                              tgt_code=code,
                                              tgt_actions=hyp.action_infos,
                                              tgt_ast=hyp.tree)
                    samples_temp.append(sampled_example)
                except:
                    pass
            if len(samples_temp)<sample_size/2.0: #valid samples too little, skip this example
                continue
            sampled_examples += samples_temp
            if len(samples_temp)<sample_size:  #valid samples less than sample size, padding by repeating valid samples
                sampled_examples += samples_temp[:sample_size-len(samples_temp)]
        sample_scores = self.encoder.score(sampled_examples)
        return sampled_examples, sample_scores

    def get_unsupervised_loss_backward(self, examples,moves):
        #compute loss for z->x->z path
        MIS_samples,sample_scores,reconstruction_scores=self.MIS_backward(examples,moves)

        encoder_loss = -reconstruction_scores
        decoder_loss = -sample_scores
        meta_data = {'samples': MIS_samples,
                     'reconstruction_scores': reconstruction_scores,
                     'encoding_scores': sample_scores}
        return encoder_loss, decoder_loss, meta_data

    def MIS_backward(self,examples,moves):

        # apply MIS on proposals of p_{\theta}(x|z)
        samples, sample_scores_with_grad,back_index = self.infer_backward(examples,moves)

        reconstruction_scores_with_grad = self.encoder.score(samples, copy=False)

        reconstruction_scores = reconstruction_scores_with_grad.data.cpu().numpy()[back_index]
        sample_scores=sample_scores_with_grad.data.cpu().numpy()[back_index]

        src_sent_var = nn_utils.to_input_variable([e.src_sent for e in samples],
                                                  self.re_prior.vocab, cuda=self.args.cuda,
                                                  append_boundary_sym=True)

        prior_scores = - self.xprior(src_sent_var).data.cpu().numpy()[back_index]
        w = (reconstruction_scores + prior_scores - sample_scores).reshape(-1, moves)


        ind = []
        cur = w[:, 0]
        ind_temp = np.arange(0, len(samples), moves)
        for i in range(1, moves):
            u = np.random.uniform(0, 1, len(w))
            accept = u < np.exp(w[:, i] - cur)
            cur = cur * (1 - accept) + w[:, i] * accept
            ind_temp = ind_temp * (1 - accept) + np.arange(i, len(samples), moves) * accept
            ind += [int(i) for i in ind_temp]

        ind=[back_index[i] for i in ind]
        MIS_samples=[samples[i] for i in ind]
        ind=Variable(torch.LongTensor(ind),volatile=True).cuda()
        sample_scores_with_grad=torch.gather(sample_scores_with_grad,dim=0,index=ind)
        reconstruction_scores_with_grad = torch.gather(reconstruction_scores_with_grad,dim=0, index=ind)
        return MIS_samples,sample_scores_with_grad,reconstruction_scores_with_grad


    def infer_backward(self, examples,sample_size):
        # get samples of p_{\theta}(x|z) on evaluating mode
        was_training = self.decoder.training
        self.decoder.eval()
        hypotheses = [self.decoder.sample(e.tgt_code,cuda=self.args.cuda) for e in examples]
        if len(hypotheses) == 0:
            raise ValueError('No candidate hypotheses.')

        if was_training: self.decoder.train()
        sampled_examples = []
        for e_id, (example, hyps) in enumerate(zip(examples, hypotheses)):
            if len(hyps)!=sample_size:
                print('reinfer not valid sample size expected %d but %d'%(sample_size,len(hyps)))
            samples_temp = []
            for hyp_id, hyp in enumerate(hyps):
                if len(hyp)==0: # generated x is null
                    continue
                try:
                    sampled_example = Example(idx='%d-resample%d' % (example.idx, hyp_id),
                                              src_sent=hyp,
                                              tgt_code=example.tgt_code,
                                              tgt_actions=example.tgt_actions,
                                              tgt_ast=example.tgt_ast)
                    samples_temp.append(sampled_example)
                except:
                    pass
            if len(samples_temp)<sample_size/2.0: #valid samples too little, skip this example
                print('beam sample so little to train')
                continue
            sampled_examples += samples_temp
            if len(samples_temp)<sample_size: #valid samples less than sample size, padding by repeating valid samples
                sampled_examples += samples_temp[:sample_size-len(samples_temp)]
        index=sorted(range(len(sampled_examples)),key=lambda i: -len(sampled_examples[i].src_sent))
        back_index=[(index[i],i) for i in index]
        back_index.sort(key=lambda x:x[0])
        assert [index[x[1]] for x in back_index]==list(range(len(sampled_examples)))
        back_index=np.array([x[1] for x in back_index])
        sampled_examples=[sampled_examples[i] for i in index]
        sample_scores = self.decoder.score(sampled_examples)
        return sampled_examples, sample_scores,back_index

    def save(self, path):
        fname, ext = os.path.splitext(path)
        self.encoder.save(fname + '.encoder' + ext)
        self.decoder.save(fname + '.decoder' + ext)
        state_dict = {k: v for k, v in self.state_dict().items() if not (k.startswith('decoder') or k.startswith('encoder') or k.startswith('prior'))}

        params = {
            'args': self.args,
            'state_dict': state_dict
        }

        torch.save(params, path)

    def load_parameters(self, path):
        fname, ext = os.path.splitext(path)
        encoder_states = torch.load(fname + '.encoder' + ext, map_location=lambda storage, loc: storage)['state_dict']
        self.encoder.load_state_dict(encoder_states)

        decoder_states = torch.load(fname + '.decoder' + ext, map_location=lambda storage, loc: storage)['state_dict']
        self.decoder.load_state_dict(decoder_states)

        vae_states = torch.load(path, map_location=lambda storage, loc: storage)['state_dict']
        self.load_state_dict(vae_states, strict=False)

    def train(self):
        super(JAE, self).train()
        self.prior.eval()


class JAE_cache(nn.Module): #JAE with caching samples
    def __init__(self, encoder, decoder, zprior,xprior, args):
        super(JAE_cache, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.zprior = zprior
        self.xprior = xprior
        self.transition_system = self.encoder.transition_system
        self.args = args
    def get_unsupervised_loss(self, examples,moves):
        #compute loss for x->z->x path
        MIS_samples,sample_scores,reconstruction_scores= self.MIS(examples, moves)
        encoder_loss = -sample_scores
        decoder_loss = -reconstruction_scores
        meta_data = {'samples': MIS_samples,
                     'reconstruction_scores': reconstruction_scores,
                     'encoding_scores': sample_scores}
        return encoder_loss, decoder_loss, meta_data

    def MIS(self, examples,moves):
        # apply MIS on proposals of q_{\phi}(z|x)
        samples, sample_scores_with_grad ,picked_example = self.infer(examples, moves)

        reconstruction_scores_with_grad=self.decoder.score(samples)
        reconstruction_scores = reconstruction_scores_with_grad.data.cpu().numpy()

        sample_scores = sample_scores_with_grad.data.cpu().numpy()

        prior_scores = self.zprior([e.tgt_code for e in samples]).data.cpu().numpy()
        w = (reconstruction_scores + prior_scores - sample_scores).reshape(-1, moves+1)

        ind = []
        cur = w[:, 0]
        ind_temp = np.arange(0, len(samples), moves+1)
        for i in range(1, moves+1):
            u = np.random.uniform(0, 1, len(w))
            accept = u < np.exp(w[:, i] - cur)
            cur = cur * (1 - accept) + w[:, i] * accept
            ind_temp = ind_temp * (1 - accept) + np.arange(i, len(samples), moves+1) * accept
            ind += [int(i) for i in ind_temp]

        MIS_samples = [samples[i] for i in ind]
        assert len(MIS_samples)==len(picked_example)
        for i,example in enumerate(picked_example):
            example.cache=MIS_samples[i]
        ind = Variable(torch.LongTensor(ind), volatile=True).cuda()
        sample_scores_with_grad = torch.gather(sample_scores_with_grad, dim=0, index=ind)
        reconstruction_scores_with_grad = torch.gather(reconstruction_scores_with_grad, dim=0, index=ind)
        return MIS_samples,sample_scores_with_grad,reconstruction_scores_with_grad

    def infer(self, examples,sample_size):
        # get samples of q_{\phi}(z|x) on evaluating mode
        was_training = self.encoder.training
        self.encoder.eval()

        hypotheses = [self.encoder.parse_sample(e.src_sent) for e in examples]

        if len(hypotheses) == 0:
            raise ValueError('No candidate hypotheses.')

        if was_training: self.encoder.train()

        # some source may not have corresponding samples, so we only retain those that have sampled logical forms
        sampled_examples = []
        picked_example=[]
        for e_id, (example, hyps) in enumerate(zip(examples, hypotheses)):
            if len(hyps) != sample_size:
                print('not valid sample size expected %d but %d' % (sample_size, len(hyps)))
            samples_temp = []
            for hyp_id, hyp in enumerate(hyps):

                try:
                    code = self.transition_system.ast_to_surface_code(hyp.tree)
                    self.transition_system.tokenize_code(code)  # make sure the code is tokenizable!
                    if len(code) == 0:
                        continue
                    sampled_example = Example(idx='%d-sample%d' % (example.idx, hyp_id),
                                              src_sent=example.src_sent,
                                              tgt_code=code,
                                              tgt_actions=hyp.action_infos,
                                              tgt_ast=hyp.tree)
                    samples_temp.append(sampled_example)
                except:
                    pass
            if len(samples_temp) < sample_size / 2.0: #valid samples too little, skip this example
                continue
            if len(samples_temp) < sample_size:  #valid samples less than sample size, padding by repeating valid samples
                samples_temp += samples_temp[:sample_size - len(samples_temp)]
            assert len(samples_temp)==sample_size
            # initialize cached samples with the first sample
            if hasattr(example, 'cache'):
                sample_final = [example.cache]
            else:
                sample_final = samples_temp[:1]
            sample_final += samples_temp
            sampled_examples+=sample_final
            picked_example.append(example)
        sample_scores = self.encoder.score(sampled_examples)

        return sampled_examples, sample_scores,picked_example

    def get_unsupervised_loss_backward(self, examples, moves):
        #compute loss for z->x->z path
        MIS_samples, sample_scores, reconstruction_scores = self.MIS_backward(examples, moves)

        encoder_loss = -reconstruction_scores
        decoder_loss = -sample_scores
        meta_data = {'samples': MIS_samples,
                     'reconstruction_scores': reconstruction_scores,
                     'encoding_scores': sample_scores}
        return encoder_loss, decoder_loss, meta_data

    def MIS_backward(self, examples,moves):
        # apply MIS on proposals of p_{\theta}(x|z)
        samples, sample_scores_with_grad, back_index,picked_example = self.infer_backward(examples, moves)
        reconstruction_scores_with_grad = self.encoder.score(samples, copy=False)

        reconstruction_scores = reconstruction_scores_with_grad.data.cpu().numpy()[back_index]
        sample_scores = sample_scores_with_grad.data.cpu().numpy()[back_index]

        src_sent_var = nn_utils.to_input_variable([e.src_sent for e in samples],
                                                  self.re_prior.vocab, cuda=self.args.cuda,
                                                  append_boundary_sym=True)
        prior_scores = - self.xprior(src_sent_var).data.cpu().numpy()[back_index]
        w = (reconstruction_scores + prior_scores - sample_scores).reshape(-1, moves+1)

        ind = []
        cur = w[:, 0]
        ind_temp = np.arange(0, len(samples), moves + 1)
        for i in range(1, moves + 1):
            u = np.random.uniform(0, 1, len(w))
            accept = u < np.exp(w[:, i] - cur)
            cur = cur * (1 - accept) + w[:, i] * accept
            ind_temp = ind_temp * (1 - accept) + np.arange(i, len(samples), moves + 1) * accept
            ind += [int(i) for i in ind_temp]

        ind = [back_index[i] for i in ind]
        MIS_samples = [samples[i] for i in ind]
        for i, example in enumerate(picked_example):
            example.recache = MIS_samples[i]
        ind = Variable(torch.LongTensor(ind),volatile=True).cuda()
        sample_scores_with_grad = torch.gather(sample_scores_with_grad, dim=0, index=ind)
        reconstruction_scores_with_grad = torch.gather(reconstruction_scores_with_grad, dim=0, index=ind)
        return MIS_samples, sample_scores_with_grad, reconstruction_scores_with_grad

    def infer_backward(self, examples, sample_size):
        # get samples of p_{\theta}(x|z) on evaluating mode
        was_training = self.decoder.training
        self.decoder.eval()
        hypotheses = [self.decoder.sample(e.tgt_code, cuda=self.args.cuda) for e in examples]
        if len(hypotheses) == 0:
            raise ValueError('No candidate hypotheses.')

        if was_training: self.decoder.train()
        sampled_examples = []
        picked_example = []
        for e_id, (example, hyps) in enumerate(zip(examples, hypotheses)):
            if len(hyps) != sample_size:
                print('not valid sample size expected %d but %d' % (sample_size, len(hyps)))
            samples_temp = []
            for hyp_id, hyp in enumerate(hyps):
                if len(hyp) == 0: # generated x is null
                    continue
                try:
                    sampled_example = Example(idx='%d-resample%d' % (example.idx, hyp_id),
                                              src_sent=hyp,
                                              tgt_code=example.tgt_code,
                                              tgt_actions=example.tgt_actions,
                                              tgt_ast=example.tgt_ast)
                    samples_temp.append(sampled_example)
                except:
                    pass
            if len(samples_temp) < sample_size / 2.0: #valid samples too little, skip this example
                continue
            if len(samples_temp) < sample_size: #valid samples less than sample size, padding by repeating valid samples
                samples_temp += samples_temp[:sample_size - len(samples_temp)]
            assert len(samples_temp) == sample_size
            # initialize cached samples with the first sample
            if hasattr(example, 'recache'):
                sample_final = [example.recache]
            else:
                sample_final = samples_temp[:1]
            sample_final += samples_temp
            sampled_examples += sample_final
            picked_example.append(example)
        index = sorted(range(len(sampled_examples)), key=lambda i: -len(sampled_examples[i].src_sent))
        back_index = [(index[i], i) for i in index]
        back_index.sort(key=lambda x: x[0])
        assert [index[x[1]] for x in back_index] == list(range(len(sampled_examples)))
        back_index = np.array([x[1] for x in back_index])
        sampled_examples = [sampled_examples[i] for i in index]
        sample_scores = self.decoder.score(sampled_examples)
        return sampled_examples, sample_scores, back_index,picked_example

    def save(self, path):
        fname, ext = os.path.splitext(path)
        self.encoder.save(fname + '.encoder' + ext)
        self.decoder.save(fname + '.decoder' + ext)
        state_dict = {k: v for k, v in self.state_dict().items() if not (k.startswith('decoder') or k.startswith('encoder') or k.startswith('prior'))}

        params = {
            'args': self.args,
            'state_dict': state_dict
        }

        torch.save(params, path)

    def load_parameters(self, path):
        fname, ext = os.path.splitext(path)
        encoder_states = torch.load(fname + '.encoder' + ext, map_location=lambda storage, loc: storage)['state_dict']
        self.encoder.load_state_dict(encoder_states)

        decoder_states = torch.load(fname + '.decoder' + ext, map_location=lambda storage, loc: storage)['state_dict']
        self.decoder.load_state_dict(decoder_states)

        vae_states = torch.load(path, map_location=lambda storage, loc: storage)['state_dict']
        self.load_state_dict(vae_states, strict=False)

    def train(self):
        super(JAE_cache, self).train()
        self.prior.eval()
