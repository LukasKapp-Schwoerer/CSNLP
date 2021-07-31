import os
import argparse
from multiprocessing import cpu_count
from utils.convert_csqa import convert_to_entailment
from utils.convert_obqa import convert_to_obqa_statement
from utils.conceptnet import extract_english, construct_graph
from utils.grounding import create_matcher_patterns, ground
from utils.graph import generate_adj_data_from_grounded_concepts__use_LM
from wiktionary.embeddings import cpnet_to_wiktionary_defs, embed_wiktionary_defs

data_dir = '/scratch/lukasks/CSNLP.git/QAGNN/data_preprocessing_cpnet_wiktionary/'

input_paths = {
    'csqa': {
        'train': data_dir + 'csqa/train_rand_split.jsonl',
        'dev': data_dir + 'csqa/dev_rand_split.jsonl',
        'test': data_dir + 'csqa/test_rand_split_no_answers.jsonl',
    },
    'obqa': {
        'train': data_dir + 'obqa/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl',
        'dev': data_dir + 'obqa/OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl',
        'test': data_dir + 'obqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl',
    },
    'obqa-fact': {
        'train': data_dir + 'obqa/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl',
        'dev': data_dir + 'obqa/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl',
        'test': data_dir + 'obqa/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl',
    },
    'cpnet': {
        'csv': data_dir + 'cpnet/conceptnet-assertions-5.6.0.csv',
    },
}

output_paths = {
    'cpnet': {
        'csv': data_dir + 'cpnet/conceptnet.en.csv',
        'vocab': data_dir + 'cpnet/concept.txt',
        'patterns': data_dir + 'cpnet/matcher_patterns.json',
        'unpruned-graph': data_dir + 'cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': data_dir + 'cpnet/conceptnet.en.pruned.graph',
        'wiktionary-definitions': data_dir + 'cpnet/concept_defs.npy',
        'wiktionary-embeddings': data_dir + 'cpnet/concept_emb.npy',
    },
    'csqa': {
        'statement': {
            'train': data_dir + 'csqa/statement/train.statement.jsonl',
            'dev': data_dir + 'csqa/statement/dev.statement.jsonl',
            'test': data_dir + 'csqa/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': data_dir + 'csqa/grounded/train.grounded.jsonl',
            'dev': data_dir + 'csqa/grounded/dev.grounded.jsonl',
            'test': data_dir + 'csqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': data_dir + 'csqa/graph/train.graph.adj.pk',
            'adj-dev': data_dir + 'csqa/graph/dev.graph.adj.pk',
            'adj-test': data_dir + 'csqa/graph/test.graph.adj.pk',
        },
    },
    'obqa': {
        'statement': {
            'train': data_dir + 'obqa/statement/train.statement.jsonl',
            'dev': data_dir + 'obqa/statement/dev.statement.jsonl',
            'test': data_dir + 'obqa/statement/test.statement.jsonl',
            'train-fairseq': data_dir + 'obqa/fairseq/official/train.jsonl',
            'dev-fairseq': data_dir + 'obqa/fairseq/official/valid.jsonl',
            'test-fairseq': data_dir + 'obqa/fairseq/official/test.jsonl',
        },
        'grounded': {
            'train': data_dir + 'obqa/grounded/train.grounded.jsonl',
            'dev': data_dir + 'obqa/grounded/dev.grounded.jsonl',
            'test': data_dir + 'obqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': data_dir + 'obqa/graph/train.graph.adj.pk',
            'adj-dev': data_dir + 'obqa/graph/dev.graph.adj.pk',
            'adj-test': data_dir + 'obqa/graph/test.graph.adj.pk',
        },
    },
    'obqa-fact': {
        'statement': {
            'train': data_dir + 'obqa/statement/train-fact.statement.jsonl',
            'dev': data_dir + 'obqa/statement/dev-fact.statement.jsonl',
            'test': data_dir + 'obqa/statement/test-fact.statement.jsonl',
            'train-fairseq': data_dir + 'obqa/fairseq/official/train-fact.jsonl',
            'dev-fairseq': data_dir + 'obqa/fairseq/official/valid-fact.jsonl',
            'test-fairseq': data_dir + 'obqa/fairseq/official/test-fact.jsonl',
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common'], choices=['common', 'wiktionary', 'csqa', 'hswag', 'anli', 'exp', 'scitail', 'phys', 'socialiqa', 'obqa', 'obqa-fact', 'make_word_vocab'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['unpruned-graph'], False)},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['pruned-graph'], True)},
            {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
            {'func': cpnet_to_wiktionary_defs, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['wiktionary-definitions'])},
            {'func': embed_wiktionary_defs, 'args': (output_paths['cpnet']['wiktionary-definitions'], output_paths['cpnet']['wiktionary-embeddings'])}
        ],
        'wiktionary': [
            {'func': cpnet_to_wiktionary_defs, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['wiktionary-definitions'])},
            {'func': embed_wiktionary_defs, 'args': (output_paths['cpnet']['wiktionary-definitions'], output_paths['cpnet']['wiktionary-embeddings'])}
        ],
        'csqa': [
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['train'], output_paths['csqa']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['test'], output_paths['csqa']['statement']['test'])},
            {'func': ground, 'args': (output_paths['csqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-test'], args.nprocs)},
        ],

        'obqa': [
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['train'], output_paths['obqa']['statement']['train'], output_paths['obqa']['statement']['train-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['dev'], output_paths['obqa']['statement']['dev'], output_paths['obqa']['statement']['dev-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['test'], output_paths['obqa']['statement']['test'], output_paths['obqa']['statement']['test-fairseq'])},
            {'func': ground, 'args': (output_paths['obqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-test'], args.nprocs)},
        ],
    }

    for rt in args.run: # default here is only common TODO: @ team, do we even preprocess th rest then?
        
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
    # pass
