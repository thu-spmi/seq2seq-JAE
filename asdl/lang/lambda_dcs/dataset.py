# coding=utf-8
from __future__ import print_function

import sys
import numpy as np
import pickle

from asdl.hypothesis import Hypothesis
from asdl.lang.py.dataset import get_action_infos
from asdl.transition_system import *
from components.dataset import Example
from components.vocab import VocabEntry, Vocab
from .lambda_dcs_transition_system import *
from .logical_form import *


def load_dataset(transition_system, dataset_file):
    examples = []
    for idx, line in enumerate(open(dataset_file)):
        src_query, tgt_code = line.strip().split('\t')

        src_query_tokens = src_query.split(' ')

        lf = parse_lambda_expr(tgt_code)
        gold_source = lf.to_string()
        assert gold_source == tgt_code
        tgt_ast = logical_form_to_ast(grammar, lf)
        reconstructed_lf = ast_to_logical_form(tgt_ast)
        assert lf == reconstructed_lf

        tgt_actions = transition_system.get_actions(tgt_ast)

        # sanity check
        hyp = Hypothesis()
        for action in tgt_actions:
            assert action.__class__ in transition_system.get_valid_continuation_types(hyp)
            if isinstance(action, ApplyRuleAction):
                assert action.production in transition_system.get_valid_continuating_productions(hyp)
            hyp = hyp.clone_and_apply_action(action)

        assert hyp.frontier_node is None and hyp.frontier_field is None

        src_from_hyp = transition_system.ast_to_surface_code(hyp.tree)
        assert src_from_hyp == gold_source

        tgt_action_infos = get_action_infos(src_query_tokens, tgt_actions)

        print(idx)
        example = Example(idx=idx,
                          src_sent=src_query_tokens,
                          tgt_actions=tgt_action_infos,
                          tgt_code=gold_source,
                          tgt_ast=tgt_ast,
                          meta=None)

        examples.append(example)

    return examples


def prepare_dataset():
    vocab_freq_cutoff = 1
    grammar = ASDLGrammar.from_text(open('asdl/lang/lambda_dcs/lambda_asdl.txt').read())
    transition_system = LambdaCalculusTransitionSystem(grammar)

    train_set = load_dataset(transition_system, 'data/atis/train.txt')
    dev_set = load_dataset(transition_system, 'data/atis/dev.txt')
    test_set = load_dataset(transition_system, 'data/atis/test.txt')

    # generate vocabulary
    src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_set], size=5000, freq_cutoff=vocab_freq_cutoff)

    primitive_tokens = [map(lambda a: a.action.token,
                            filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                        for e in train_set]

    primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=5000, freq_cutoff=0)

    # generate vocabulary for the code tokens!
    code_tokens = [transition_system.tokenize_code(e.tgt_code, mode='decoder') for e in train_set]
    code_vocab = VocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=0)

    vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=code_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    action_len = [len(e.tgt_actions) for e in chain(train_set, dev_set, test_set)]
    print('Max action len: %d' % max(action_len), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
    print('Actions larger than 100: %d' % len(filter(lambda x: x > 100, action_len)), file=sys.stderr)

    pickle.dump(train_set, open('data/atis/train.bin', 'w'))
    pickle.dump(dev_set, open('data/atis/dev.bin', 'w'))
    pickle.dump(test_set, open('data/atis/test.bin', 'w'))
    pickle.dump(vocab, open('data/atis/vocab.bin', 'w'))


if __name__ == '__main__':
    grammar = ASDLGrammar.from_text(open('asdl/lang/lambda_dcs/lambda_asdl.txt').read())
    transition_system = LambdaCalculusTransitionSystem(grammar)
    # load_dataset(transition_system, 'data/atis/train.txt')
    prepare_dataset()
