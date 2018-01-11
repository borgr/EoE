import os
import sys
import numpy as np
import nltk
from scipy.stats.stats import pearsonr, spearmanr

from utils import *
sys.path.append(ASSESS_DIR)
sys.path.append(ASSESS_DIR + '/m2scorer/scripts')
from assess_learner_language.rank import BLEU_score
from create_db import create_ranks, iterate_chains, ANNOTATION_FILE
os.path.join(PROJECT, "data", "all_references.m2")


def choose_uniformely(lst):
    return lst[np.random.randint(len(lst))]


def choose_source_per_chain(ranks, chooser=choose_uniformely):
    return [apply_changes(*chooser(chain)) for chain in iterate_chains(ranks)]


def choose_source_per_sentence(ranks, chooser=choose_uniformely):
    sources = []
    for sentence_chains in ranks:
        annot_chains = np.random.choice(sentence_chains)
        chain = np.random.choice(annot_chains)
        sources.append(chooser(chain))
    return sources


def extract_references(ids, reference_files):
    res = np.asarray([list(get_lines_from_file(fl, ids))
                      for fl in reference_files]).transpose()
    return res


def extract_references_per_chain(ranks, ids, reference_files):
    ids = [tple[1] for tple in iterate_chains(ranks, ids)]
    return extract_references(ids, reference_files)


def score(sources, all_references, ranks, sentence_measure, corpus_measure=None):
    """scores the corpus by the measure of choice. returns the score and the rankings
       measure - a function that gets (source, references_iterable, sentence) and a returns a number
       if a corpus measure is given, all sources\references_iterables\sentences would be passed to it as an iterable instead of being evaluated one by one by sentence_measure."""
    if corpus_measure is not None:
        scores_it = "not implemented"
        raise "unimplemented yet"
    scores = []
    for source, references, chain in zip(sources, all_references, iterate_chains(ranks)):
        chain_scores = []
        for sentence in chain:
            if corpus_measure is None:
                score = sentence_measure(
                    source, references, apply_changes(*sentence))
            else:
                score = next(scores_it)
            chain_scores.append(score)
        scores.append(chain_scores)
    return scores


def ranks_to_scores(ranks):
    scores = []
    for chain in iterate_chains(ranks):
        chain_scores = list(range(len(chain)))
        scores.append(chain_scores)
    return scores


def assess_measures(measures, ranks, ids, reference_files, cache=None):
    sources = choose_source_per_chain(ranks)
    references = extract_references_per_chain(
        ranks, ids, reference_files)
    human_flatten_ranks = list(traverse_chains(ranks_to_scores(ranks)))
    for measure_details in measures:
        name = measure_details[0]
        sentence_measure = measure_details[1]
        corpus_measure = measure_details[2]
        print(name)
        measure_score = score(sources, references, ranks,
                              sentence_measure, corpus_measure)
        measure_flatten_score = list(traverse_chains(measure_score))
        assert(len(measure_flatten_score) == len(human_flatten_ranks))
        print("sentence level correlations:")
        # print(list(zip(range(2),human_flatten_ranks, measure_score)))
        pearson = pearsonr(human_flatten_ranks, measure_flatten_score)
        spearman = spearmanr(human_flatten_ranks, measure_flatten_score)
        print("pearson (val, P-val):",pearson[0], pearson[1])
        print("spearman (val, P-val):",spearman[0], spearman[1])


def main():
    bn = os.path.join(REFERENCE_DIR, "BN")
    bn_refs = []
    bn_refs += [bn + str(i) for i in [1, 2, 4, 5, 6, 8, 9, 10]]
    nucleA_ref = bn + "7"
    nucleB_ref = bn + "3"
    nucle_refs = [nucleA_ref, nucleB_ref]
    reference_files = bn_refs
    max_permutations = 1
    filename = str(max_permutations) + "_" + "rank" + ".json"
    ids_filename = os.path.join(CACHE_DIR,  "id" + filename)
    ranks_filename = os.path.join(CACHE_DIR,  "rank" + filename)
    ranks, ids = create_ranks(ANNOTATION_FILE, max_permutations,
                              ranks_out_file=ranks_filename, ids_out_file=ids_filename)
    measures = []
    # measures format
    # measures.append((name, 
    #                  function(source, references,system_sentence):score or None, 
    #                  function(sources,all_references,sentences):score iterable or None)
    measures.append(("BLEU", lambda so, r, sy: BLEU_score(
        so, r, sy, 4, nltk.translate.bleu_score.SmoothingFunction().method3, lambda x: x), None))
    assess_measures(measures, ranks, ids, reference_files)
    # bleu_rank=score(sources, references, ranks, lambda so, r, sy: BLEU_score(
    # so, r, sy, 4, nltk.translate.bleu_score.SmoothingFunction().method3,
    # lambda x: x))
    # print(list(traverse_chains(human_ranks)))


if __name__ == '__main__':
    main()
