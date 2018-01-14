import os
import sys
import numpy as np
import nltk
from scipy.stats.stats import pearsonr, spearmanr
import six

from utils import *
sys.path.append(ASSESS_DIR)
sys.path.append(ASSESS_DIR + '/m2scorer/scripts')
from assess_learner_language.rank import BLEU_score, gleu_scores
from create_db import create_ranks, create_corpora, ANNOTATION_FILE
os.path.join(PROJECT, "data", "all_references.m2")

SENTENCE_CACHE_DIR = os.path.join(CACHE_DIR, "sentence")
CORPUS_CACHE_DIR = os.path.join(CACHE_DIR, "corpus")
if not os.path.isdir(SENTENCE_CACHE_DIR):
    os.makedirs(SENTENCE_CACHE_DIR)
if not os.path.isdir(CORPUS_CACHE_DIR):
    os.makedirs(CORPUS_CACHE_DIR)


def choose_source_per_chain(ranks, chooser=choose_uniformely):
    return [chooser(chain)[0] for chain in iterate_chains(ranks)]


def choose_source_per_sentence(ranks, chooser=choose_uniformely):
    sources = []
    for sentence_chains in ranks:
        annot_chains = choose_uniformely(sentence_chains)
        chain = choose_uniformely(annot_chains)
        sources.append(chooser(chain))
    return sources


def extract_references(ids, reference_files):
    res = np.asarray([list(get_lines_from_file(fl, ids))
                      for fl in reference_files]).transpose()
    return res


def extract_references_per_chain(ranks, ids, reference_files):
    ids = [tple[1] for tple in iterate_chains(ranks, ids)]
    return extract_references(ids, reference_files)


def score_corpus(sources, all_references, corpus, sentence_measure, corpus_measure=None, cache_file=None, force=False):
    if not force and cache_file is not None:
        if os.path.isfile(cache_file):
            # print("reading measure from cache", cache_file)
            return load_object_by_ext(cache_file)
    if not isinstance(corpus[0], six.string_types):
        corpus = [sent[0] for sent in corpus]
    if not isinstance(sources[0], six.string_types):
        sources = [sent[0] for sent in sources]
    if corpus_measure is not None:
        all_references = np.array(all_references).transpose()
        scores = corpus_measure(sources, all_references, corpus)
    else:
        scores = [sentence_measure(source, references, sentence)
                  for source, references, sentence in zip(sources, all_references, corpus)]
    if cache_file is not None:
        save_object_by_ext(scores, cache_file)
    return scores


def score_corpora(sources, all_references, corpora, sentence_measure, corpus_measure=None, cache_files=None, force=False):
    if cache_files is None:
        cache_files = [None] * len(corpora)
    scores = [score_corpus(sources, all_references, corpus, sentence_measure,
                           corpus_measure, cache_file, force) for corpus, cache_file in zip(corpora, cache_files)]
    return scores


def score(sources, all_references, ranks, sentence_measure, corpus_measure=None, cache_file=None, force=False):
    """scores the corpus by the measure of choice. returns the score and the rankings
       measure - a function that gets (source, references_iterable, sentence) and a returns a number
       if a corpus measure is given, all sources\references_iterables\sentences would be passed to it as an iterable instead of being evaluated one by one by sentence_measure."""
    if not force and cache_file is not None:
        if os.path.isfile(cache_file):
            # print("reading measure from cache", cache_file)
            return load_object_by_ext(cache_file)

    if corpus_measure is not None:
        corpus_sources = []
        corpus_references = []
        sentences = []
        for source, references, chain in zip(sources, all_references, iterate_chains(ranks)):
            for sentence in chain:
                corpus_sources.append(source)
                corpus_references.append(references)
                sentences.append(sentence[0])
        corpus_references = np.array(corpus_references).transpose()
        scores_it = corpus_measure(
            corpus_sources, corpus_references, sentences)
        try:
            scores_it = iter(scores_it)
        except TypeError:
            raise "corpus measure should be None or a function that returns an iterator or an iterable"
    scores = []
    for source, references, chain in zip(sources, all_references, iterate_chains(ranks)):
        chain_scores = []
        for sentence in chain:
            if corpus_measure is None:
                score = sentence_measure(
                    source, references, sentence[0])
            else:
                score = next(scores_it)
            chain_scores.append(score)
        scores.append(chain_scores)
    if cache_file is not None:
        save_object_by_ext(scores, cache_file)
    return scores


def sentence_length(sent):
    return len(sent.split())


def ranks_to_scores(ranks):
    scores = []
    for chain in iterate_chains(ranks):
        chain_scores = range(len(chain))
        # make perfect sentences with 0 score
        chain_scores = reversed(chain_scores)
        chain_scores = np.fromiter(chain_scores, np.float, len(chain))
        chain_scores = 1 - (chain_scores / sentence_length(chain[0][0]))
        scores.append(chain_scores)
    return scores


def assess_measures(measures, ranks, ids, corpora, corpus_ids, reference_files, corpora_names=None, corpora_scores=None, cache=None, force=False):
    """ runs all assessments on a given measure
    measures comply to the format:
    (name,
    function(source, references,system_sentence):score or None,
    function(sources,all_references,sentences):score iterable or None))
    ranks - 
    ids - 
    reference_files - 
    cache - basename to save caches, None to avoid caching altogether, 
            the basename is changed using the name of the measures, 
            duplicate names should hence be avoided
    force - whether to force recalculation and overwrite cache
    """
    if corpora_scores is None:
        corpora_scores = list(range(len(corpora)))
    sources = choose_source_per_chain(ranks)
    references = extract_references_per_chain(
        ranks, ids, reference_files)
    print("choosing first corpus as source")
    corpus_source = corpora[0]
    corpus_references = extract_references(corpus_ids, reference_files)
    human_flatten_ranks = list(traverse_chains(ranks_to_scores(ranks)))
    for measure_details in measures:
        name = measure_details[0]
        sentence_measure = measure_details[1]
        corpus_measure = measure_details[2]
        if cache is not None:
            cache_file = os.path.join(os.path.dirname(
                cache), name + "_" + os.path.basename(cache))
            if corpora_names is not None:
                assert len(corpora_names) == len(
                    corpora), "each corpus must be named, to name no corpus pass None,\
                     to skip caching some of the corpora pass None instead of names in the needed indexes"
                cache_names = []
                for corpus_name in corpora_names:
                    if corpus_name is None:
                        print("skipping corpus cache")
                        cache_names.append(None)
                    else:
                        cache_names.append(os.path.join(os.path.dirname(cache), str(
                            corpus_name) + "_corpus_" + name + "_" + os.path.basename(cache)))
            else:
                cache_names = None
        print(name)
        measure_corpora_scores = score_corpora(
            corpus_source, corpus_references, corpora, sentence_measure, corpus_measure, cache_names, force)
        aggregated_corpora_scores = np.mean(measure_corpora_scores, 1)
        measure_score = score(sources, references, ranks,
                              sentence_measure, corpus_measure, cache_file=cache_file, force=force)
        measure_flatten_score = list(traverse_chains(measure_score))
        assert(len(measure_flatten_score) == len(human_flatten_ranks))
        print("corpus level correlations:")
        pearson = pearsonr(corpora_scores, aggregated_corpora_scores)
        spearman = spearmanr(corpora_scores, aggregated_corpora_scores)
        print("pearson (val, P-val):", pearson[0], pearson[1])
        print("spearman (val, P-val):", spearman[0], spearman[1])
        print("corpus scores for manual analysis, together with their human scores")
        print(np.array(list(zip(corpora_scores, aggregated_corpora_scores))).transpose())

        print("sentence level correlations:")
        # print(list(zip(range(2),human_flatten_ranks, measure_score)))
        pearson = pearsonr(human_flatten_ranks, measure_flatten_score)
        spearman = spearmanr(human_flatten_ranks, measure_flatten_score)
        print("pearson (val, P-val):", pearson[0], pearson[1])
        print("spearman (val, P-val):", spearman[0], spearman[1])
        print("some scores for manual analysis, together with their human ranks")
        print(np.array(list(zip(human_flatten_ranks, measure_flatten_score)))[
              :7].transpose())


def _glue_wrapper(sources, references, sentences):
    res = gleu_scores(sources, references, [sentences])[1]
    return [float(mean) for mean, std, confidence_interval in res]


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

    # sentence level inits
    ids_filename = os.path.join(SENTENCE_CACHE_DIR,  "id" + filename)
    ranks_filename = os.path.join(SENTENCE_CACHE_DIR,  "rank" + filename)
    ranks, ids = create_ranks(ANNOTATION_FILE, max_permutations,
                              ranks_out_file=ranks_filename, ids_out_file=ids_filename)
    cache_scores = os.path.join(SENTENCE_CACHE_DIR,  "score" + filename)

    # corpus level inits
    corpora_ids_filename = os.path.join(
        CORPUS_CACHE_DIR,  "corora_id" + filename)
    corpora_basename = os.path.join(CORPUS_CACHE_DIR,  "corpus" + filename)
    corpus_mean_corrections = [0, 2, 4, 6, 8, 10]
    exact_prob_vars = corpus_mean_corrections
    bin_prob_vars = [binomial_parameters_by_mean_and_var(
        i, 0.9) for i in corpus_mean_corrections]

    prob_vars = bin_prob_vars
    # prob_vars = exact_prob_vars
    if prob_vars == bin_prob_vars:
        corpora_names = [str(x) + "," + str(y) for x, y in prob_vars]
    elif prob_vars == exact_prob_vars:
        corpora_names = [str(x) for x in prob_vars]
    corpora, corpus_ids = create_corpora(ANNOTATION_FILE, prob_vars,
                                         corpora_basename=corpora_basename, ids_out_file=corpora_ids_filename)

    force = True
    force = False
    if force:
        print("Forcing recalculation")
    measures = []
    # measures format
    # measures.append((name,
    #                  function(source, references,system_sentence):score or None,
    # function(sources,all_references,sentences):score iterable or None)
    measures.append(("GLEU", None, _glue_wrapper))
    measures.append(("BLEU", lambda so, r, sy: BLEU_score(
        so, r, sy, 4, nltk.translate.bleu_score.SmoothingFunction().method3, lambda x: x), None))
    assess_measures(measures, ranks, ids, corpora, corpus_ids,
                    reference_files, corpora_names, corpus_mean_corrections, cache_scores, force)
    # bleu_rank=score(sources, references, ranks, lambda so, r, sy: BLEU_score(
    # so, r, sy, 4, nltk.translate.bleu_score.SmoothingFunction().method3,
    # lambda x: x))
    # print(list(traverse_chains(human_ranks)))


if __name__ == '__main__':
    main()
