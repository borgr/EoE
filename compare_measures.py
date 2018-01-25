import os
import sys
import shutil
from itertools import repeat
import multiprocessing
import argparse
from datetime import datetime
import random

import numpy as np
import nltk
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
import six
import distance
POOL_SIZE = multiprocessing.cpu_count()
DEBUG = False
from utils import *
sys.path.append(ASSESS_DIR)
sys.path.append(ASSESS_DIR + '/m2scorer/scripts')
from assess_learner_language.m2scorer import m2scorer
from assess_learner_language.rank import BLEU_score, SARI_score, SARI_max_score
from assess_learner_language.rank import sentence_m2, gleu_scores, Imeasure_scores
from assess_learner_language.rank import grammaticality_score, semantics_score, reference_less_score
from assess_learner_language.rank import ucca_parse_sentences, create_one_sentence_files, parse_location
from assess_learner_language.annalyze_crowdsourcing import convert_correction_to_m2
from create_db import create_ranks, create_corpora, ANNOTATION_FILE
from assess_learner_language.errant import parallel_to_m2 as p2m2
os.path.join(PROJECT, "data", "all_references.m2")

SOURCE = "source"
REF = "ref"
RAND = "rand"

# measure attributes
NAME = "name"
SENTENCE_MEASURE = "sent_measure"
CORPUS_MEASURE = "corpus_measure"
CORPUS_SCORER = "corpus_scorer"
EDIT_BASED = "edit_based"
PREPROCESS = "preprocess_func"
PREPROCESS_CORPUS = "preprocess_corpus_func"

CHANGES_LOC = 1
SENTENCE_LOC = 0


def choose_source_per_chain(ranks, chooser=choose_uniformely):
    return [chooser(chain)[SENTENCE_LOC] for chain in iterate_chains(ranks)]


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


_convert2edits_cache = {}


def convert2edits(sources, all_references, cache_file=None):
    hashed = list_to_hashable(sources), list_to_hashable(all_references)
    if hashed in _convert2edits_cache:
        return _convert2edits_cache[hashed]
    # res = []
    # for source, references in zip(sources, all_references):
    #     source = source[0]
    #     res.append("S " + source + "\n")
    #     for i, reference in enumerate(references):
    #         addition = convert_correction_to_m2(source, reference, i)
    #         if not addition:
    #             addition = [
    #                 "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||" + str(i) + "\n"]
    #         res += addition
    # sources = [source[0] for source in sources]
    res = p2m2.parallel_to_m2(sources, all_references)
    if cache_file is None:
        cache_file = os.path.join(
            CACHE_DIR, datetime.now().strftime('edits_%Y_%m_%d_%H_%M_%S.m2'))
    save_object_by_ext(res, cache_file)
    _, res = m2scorer.load_annotation(cache_file)
    os.remove(cache_file)
    _convert2edits_cache[hashed] = res
    return res


def score_corpus(sources, all_references, corpus, sentence_measure, corpus_measure=None, edit_based=False, cache_file=None, force=False):
    # if DEBUG:
    #     sources = sources[:2]
    #     corpus = corpus[:2]
    if not force and cache_file is not None:
        if os.path.isfile(cache_file):
            print("reading measure from cache", cache_file)
            return load_object_by_ext(cache_file)
    if not isinstance(corpus[0], six.string_types):
        corpus = [sent[SENTENCE_LOC] for sent in corpus]
    if not isinstance(sources[0], six.string_types):
        sources = [sent[SENTENCE_LOC] for sent in sources]
    if edit_based:
        all_references = convert2edits(sources, all_references)
    if corpus_measure is not None:
        all_references = np.array(all_references).transpose()
        scores = corpus_measure(sources, all_references, corpus)
    else:
        scores = [sentence_measure(source, references, sentence)
                  for source, references, sentence in zip(sources, all_references, corpus)]
    if cache_file is not None:
        save_object_by_ext(scores, cache_file)
    return scores


def score_corpora(sources, all_references, corpora, sentence_measure, corpus_measure=None, edit_based=False, cache_files=None, force=False):
    print("score corpora")
    if cache_files is None:
        cache_files = [None] * len(corpora)
    pool = multiprocessing.Pool(POOL_SIZE)
    scores = pool.starmap(score_corpus, zip(repeat(sources), repeat(all_references), corpora, repeat(sentence_measure),
                                            repeat(corpus_measure), repeat(edit_based), cache_files, repeat(force)))
    pool.close()
    pool.join()
    # scores = [score_corpus(sources, all_references, corpus, sentence_measure,
    # corpus_measure, cache_file, force) for corpus, cache_file in
    # zip(corpora, cache_files)]
    return scores


def measure_corpora(sources, all_references, corpora, sentence_measure, corpus_measure=None, edit_based=False, cache_files=None, corpora_scores=None, force=False):
    aggregated_corpora_scores = score_corpora(
        sources, all_references, corpora, None, corpus_measure, edit_based, cache_files, force)
    aggregated_corpora_scores = np.squeeze(aggregated_corpora_scores)
    if corpora_scores is None:
        return aggregated_corpora_scores
    if aggregated_corpora_scores.size == len(corpora_scores):
        aggregated_corpora_scores = aggregated_corpora_scores.ravel()
    elif len(aggregated_corpora_scores.shape) > 1:
        if len(corpora_scores) in aggregated_corpora_scores.shape:
            idx = 1 - \
                aggregated_corpora_scores.shape.index(len(corpora_scores))
            aggregated_corpora_scores = np.mean(aggregated_corpora_scores, idx)
            print("Aggregeted using mean over index", idx)
        else:
            raise ValueError("Warning: corpus_measure returned an array of shape",
                             aggregated_corpora_scores.shape, "instead of a list of length", len(corpora_scores))

    return aggregated_corpora_scores


def score_sentences(sources, all_references, ranks, sentence_measure, corpus_measure=None, edit_based=False, cache_file=None, force=False):
    """scores the corpus by the measure of choice. returns the score and the rankings
       measure - a function that gets (source, references_iterable, sentence) and a returns a number
       if a corpus measure is given, all sources\references_iterables\sentences would be passed to it as an iterable instead of being evaluated one by one by sentence_measure."""
    if not force and cache_file is not None:
        if os.path.isfile(cache_file):
            print("reading measure from cache", cache_file)
            return load_object_by_ext(cache_file)
    if edit_based:
        all_references = convert2edits(sources, all_references)
    if corpus_measure is not None:
        corpus_sources = []
        corpus_references = []
        sentences = []
        for source, references, chain in zip(sources, all_references, iterate_chains(ranks)):
            for sentence in chain:
                corpus_sources.append(source)
                corpus_references.append(references)
                sentences.append(sentence[SENTENCE_LOC])
        corpus_references = np.array(corpus_references).transpose()
        print("corpus")
        scores_it = corpus_measure(
            corpus_sources, corpus_references, sentences)
        # print(scores_it)
        try:
            scores_it = iter(scores_it)
        except TypeError:
            print(scores_it)
            raise ValueError(
                "corpus measure should be None or a function that returns an iterator or an iterable")
    scores = []
    max_chain_len = 0
    max_sen_len = 0
    for source, references, chain in zip(sources, all_references, iterate_chains(ranks)):
        if max_chain_len < len(chain):
            max_chain_len = len(chain)
        if max_sen_len < len(chain[0][SENTENCE_LOC]):
            max_sen_len = len(chain[0][SENTENCE_LOC])
        chain_scores = []
        for sentence in chain:
            if corpus_measure is None:
                score = sentence_measure(
                    source, references, sentence[SENTENCE_LOC])
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
    for sentence_chains in ranks:
        original_word_count = len(sentence_chains[0][0][SENTENCE_LOC])
        # original_score is as good as the minimum amount of edis needed to
        # make it a reference normalized by its length
        original_score = 1 - min((len(chain)
                                  for chain in sentence_chains)) / original_word_count
        # ref_score is perfect
        ref_score = 1
        for chain in sentence_chains:
            chain_scores = np.linspace(original_score, ref_score, len(chain))

            # # sentence score is 1 - number of changes from ref/original length
            # chain_scores = range(len(chain))
            # chain_scores = reversed(chain_scores)
            # chain_scores = np.fromiter(chain_scores, np.float, len(chain))
            # chain_scores = 1 - (chain_scores / sentence_length(chain[0][0]))
            scores.append(chain_scores)
    return scores


def extract_matches(sources, references, ranks, human_scores, measure_score):
    """ returns a list of tuples of 
    (source, references, 
    first sentence, first measure, first human,
    second sentence, second measure_score, second human_score)
    where the measure rises together with the human score and another where it is not"""
    matches = []
    mismatches = []
    for source, reference, chain, human_chain, measure_chain, in zip(sources, references, iterate_chains(ranks), human_scores, measure_score):
        last_measure = None
        for sentence, human_score, measure_score in zip(chain, human_chain, measure_chain):
            if last_measure is None:
                last_measure = measure_score
                last_human = human_score
                last_sentence = sentence
            else:
                if (last_measure > measure_score and last_human > human_score) or (last_measure < measure_score and last_human < human_score):
                    cur = matches
                else:
                    cur = mismatches
                cur.append((source, references, last_sentence, last_measure,
                            last_human, sentence, measure_score, human_score))
    return matches, mismatches


def score_changes_per_type(ranks, scores):
    difs_by_type = {}
    for chain, scores_chain, in zip(iterate_chains(ranks), scores):
        last_score = None
        for sentence, score in zip(chain, scores_chain):
            if last_score is None:
                last_score = score
            else:
                # Because of the chain structure, the last error type corrected
                # is the only change from last sentence
                # sentence[changes][last correction][correction type]
                error_type = sentence[CHANGES_LOC][-1][2]
                if error_type not in difs_by_type:
                    difs_by_type[error_type] = []
                difs_by_type[error_type].append(score - last_score)
    return difs_by_type


def assess_measures(measures, ranks, ids, corpora, corpus_ids, reference_files, choose_corpus_source=lambda x: x[SENTENCE_LOC], corpora_names=None, corpora_scores=None, manual_analysis_num=True, matches_num=0, cache=None, force=False):
    """ runs all assessments on a given measure
    measures comply to the format:
    (name,
    function(source, references,system_sentence):score or None,
    function(sources,all_references,sentences):score iterable or None)
    function(sources,all_references,sentences):score or None))
    measures contain:
    measure name
    function that returns a score of a sentence or that returns all scores for a list of sentences
    possibly a function that returns one score for a list of sentences
    ranks - 
    ids - 
    corpora - list of corpora, whether in text form or in tuples of text and the changes made on it from the original
    corpus_ids - lines used in all the corpora
    reference_files - 
    corpora_names - list of names to be appended to the cache filenamename
    corpora_scores - human score for each corpus
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
    if cache is not None:
        corpus_source_file = os.path.join(os.path.dirname(
            cache), "_corpsource" + "_" + os.path.basename(cache))
        if os.path.isfile(corpus_source_file):
            corpus_source = load_object_by_ext(corpus_source_file)
        else:
            corpus_source = choose_corpus_source(corpora)
            corpus_source = [source[SENTENCE_LOC] for source in corpus_source]
            save_object_by_ext(corpus_source, corpus_source_file)
    corpus_references = extract_references(corpus_ids, reference_files)
    if DEBUG:
        sources = sources[:2]
        references = references[:2]
        ranks = ranks[:2]
        corpus_source = corpus_source[:2]
        corpus_references = corpus_references[:2]
        corpora = [x[:2] for x in corpora]
    human_scores = ranks_to_scores(ranks)
    print("Overall corpus statistics:")
    print("Mean human change per type:")
    for error_type, changes in score_changes_per_type(ranks, human_scores).items():
        print(error_type, ": ", np.mean(changes), sep="")
    # number of sentences that differ by exactly one correction from a
    # specific type
    print()
    print("Corrections of each type:")
    for error_type, changes in score_changes_per_type(ranks, human_scores).items():
        print(error_type, ": ", len(changes), sep="")
    human_ranks = [np.argsort(x) for x in human_scores]
    human_flatten_scores = list(traverse_chains(human_scores))
    # check each measure
    for measure_details in measures:
        name = from_measure(measure_details, NAME)
        sentence_measure = from_measure(measure_details, SENTENCE_MEASURE)
        corpus_measure = from_measure(measure_details, CORPUS_MEASURE)
        corpus_scorer = from_measure(measure_details, CORPUS_SCORER)
        edit_based = from_measure(measure_details, EDIT_BASED)
        print()
        print(name)
        preprocess_sentence_level_func = from_measure(
            measure_details, PREPROCESS)
        preprocess_corpus_level_func = from_measure(
            measure_details, PREPROCESS_CORPUS)
        if cache is not None:
            cache_file = os.path.join(os.path.dirname(
                cache), name + "_" + os.path.basename(cache))
            if corpora_names is not None:
                assert len(corpora_names) == len(
                    corpora), "Each corpus must be named, to name no corpus pass None,\
                     to skip caching some of the corpora pass None instead of names in the needed indexes"
                cache_files = []
                for corpus_name in corpora_names:
                    if corpus_name is None:
                        print("skipping corpus cache")
                        cache_files.append(None)
                    else:
                        cache_files.append(os.path.join(os.path.dirname(cache), str(
                            corpus_name) + "_corpus_" + name + "_" + os.path.basename(cache)))
            else:
                cache_files = None
        p_sources, p_references, p_ranks = preprocess_sentence_level_func(
            sources, references, ranks)
        p_corpus_source, p_corpus_references, p_corpora = preprocess_corpus_level_func(
            corpus_source, corpus_references, corpora)
        measure_score = score_sentences(p_sources, p_references, p_ranks,
                                        sentence_measure, corpus_measure, edit_based, cache_file=cache_file, force=force)
        if corpus_scorer is None:
            measure_corpora_scores = score_corpora(
                p_corpus_source, p_corpus_references, p_corpora, sentence_measure, corpus_measure, edit_based, cache_files, force)
            aggregated_corpora_scores = np.mean(measure_corpora_scores, 1)
        else:
            aggregated_corpora_scores = measure_corpora(
                p_corpus_source, p_corpus_references, p_corpora, None, corpus_scorer, edit_based, cache_files, corpora_scores, force)
        measure_ranks = [np.argsort(x) for x in measure_score]
        measure_flatten_score = list(traverse_chains(measure_score))
        print("corpus level correlations:")
        print_list_statistics(
            aggregated_corpora_scores, corpora_scores, manual_analysis_num)
        if DEBUG:
            human_flatten_scores = human_flatten_scores[
                :len(measure_flatten_score)]
        print()
        print("sentence level correlations:")
        print_list_statistics(measure_flatten_score,
                              human_flatten_scores, manual_analysis_num)
        if matches_num:
            matches, mismatches = extract_matches(
                sources, references, ranks, human_scores, measure_score)
            print("Some matches for manual analysis (source, references + sentence, changes apllied, measure, human of two sentences)")
            print(matches[:matches_num])
            print("Some mismatches for manual analysis (source, references + sentence, changes apllied, measure, human of two sentences)")
            print(mismatches[:matches_num])
        print()
        print("chain level correlations:")
        # kendall = kendall_in_parts(human_scores, measure_ranks)
        kendall = kendall_partial_order_from_seq(
            human_scores, measure_ranks, traverse_ranks(ranks))
        print("Kendall's (val, P-val):", kendall[0], kendall[1])
        print("Mean score change per correction type applied:")
        for error_type, changes in score_changes_per_type(ranks, measure_score).items():
            print(error_type, ": ", np.mean(changes), sep="")


def print_list_statistics(x, y, manual_analysis_num):
    assert len(x) == len(y), " Lists were expected to be in the same length and not " + \
        str(len(y)) + ", " + str(len(x))
    pearson = pearsonr(y, x)
    print("Pearson (val, P-val):", pearson[0], pearson[1])
    # spearman = spearmanr(y, x)
    # print("Spearman (val, P-val):", spearman[0], spearman[1])
    # kendall = kendalltau(y, x)
    # print("Kendall's Tau (val, P-val):", kendall[0], kendall[1])
    if manual_analysis_num:
        print("Some scores for manual analysis, together with their corresponding ranks")
        print(np.array(list(zip(y, x)))[
              :manual_analysis_num].transpose())


class Imeasure_callable(object):

    def __init__(self, ref_file, return_per_sent_scores=True, mixmax=100, **kwargs):
        self.ref_file = ref_file
        self.return_per_sent_scores = return_per_sent_scores
        self.mixmax = mixmax
        self.bound_args = kwargs

    def get_score(self, imeasore_score):
        return imeasore_score["c"]["wacc"] * 100

    def __call__(self, sources, references, sentences, **kwargs):
        res = Imeasure_scores(sources, self.ref_file, sentences,
                              return_per_sent_scores=self.return_per_sent_scores, mixmax=self.mixmax, quiet=False, **self.bound_args, **kwargs)
        if self.return_per_sent_scores:
            res = res[1]
            res = [self.get_score(r) for r in res]
        else:
            res = self.get_score(res)
        return res


class Reference_less_callabale(object):

    def __init__(self, cache_folder, **kwargs):
        self.cache_folder = cache_folder

    def __call__(self, source, references, sentence, **kwargs):
        raise Exception("can't call abstract class")


class CombinedReference_less_callabale(Reference_less_callabale):

    def __init__(self, cache_folder, gamma, **kwargs):
        super().__init__(cache_folder)
        self.gamma = gamma

    def __call__(self, source, references, sentence, **kwargs):
        return reference_less_score(source, sentence, self.cache_folder, self.gamma)


class USim_callabale(Reference_less_callabale):

    def __call__(self, source, references, sentence, **kwargs):
        res = semantics_score(source, sentence, self.cache_folder)
        return res


class Grammaticallity_callabale(Reference_less_callabale):

    def __call__(self, source, references, sentence, **kwargs):
        res = grammaticality_score(source, sentence, self.cache_folder)
        return res


def _glue_wrapper(sources, references, sentences):
    res = gleu_scores(sources, references, [sentences])[1]
    return [float(mean) for mean, std, confidence_interval in res]


def _levenshtein_wrapper(source, references, sentence):
    return _levenshtein_score(source, sentence)


def _levenshtein_references_wrapper(source, references, sentence):
    return max((_levenshtein_score(ref, sentence) for ref in references))


def _levenshtein_score(source, sentence):
    leven = distance.levenshtein(source, sentence)
    leven = 1 if len(source) == 0 else 1 - leven / len(source)
    return leven


def _m2_wrapper(source, edits, sentence):
    return sentence_m2(source, edits, sentence)[-1]


def _bleu_wrapper(source, references, sentence):
    return BLEU_score(
        source, references, sentence, 4, nltk.translate.bleu_score.SmoothingFunction().method3, lambda x: x)


def _ibleu_wrapper(source, references, sentence, alpha=0.8):
    return _bleu_wrapper(source, references, sentence) * alpha + (1 - alpha) * _bleu_wrapper(source, [source], sentence)


def shuffle_sources(corpora):
    corpus = []
    corpora_num = len(corpora)
    corpus_len = len(corpora[0])
    for i in range(corpus_len):
        corpus.append(corpora[np.random.randint(corpora_num)][i])
    return corpus


def from_measure(measure_details, attribute):
    if attribute == NAME:
        return measure_details[0]
    if attribute == SENTENCE_MEASURE:
        return measure_details[1]
    if attribute == CORPUS_MEASURE:
        return measure_details[2]
    if attribute == CORPUS_SCORER:
        return measure_details[3]
    if attribute == EDIT_BASED:
        return measure_details[4]
    if attribute == PREPROCESS:
        return measure_details[5]
    if attribute == PREPROCESS_CORPUS:
        return measure_details[6]


def add_measure(measures, name, sentence_measure=None, corpus_measure=None, corpus_scorer=None, edit_based=False, preprocess_sentence_level=lambda x, y, z: (x, y, z), preprocess_corpus_level=lambda x, y, z: (x, y, z)):
    assert not (
        sentence_measure is None and corpus_measure is None and corpus_scorer is None)
    measures.append((name, sentence_measure, corpus_measure,
                     corpus_scorer, edit_based, preprocess_sentence_level, preprocess_corpus_level))


def sentence_input_to_sentence_list(source, references, ranks):
    return source + [reference for reference_list in references for reference in reference_list] + [
        sent[SENTENCE_LOC] for sent in traverse_ranks(ranks)]


def corpus_input_to_sentence_list(corpus_source, corpus_references, corpora):
    return corpus_source + [reference for references in corpus_references for reference in references] + [
        sent[SENTENCE_LOC] for corpus in corpora for sent in corpus]


def parse_combined_sentence(source, references, ranks, ucca_parse_dir, filename, one_sentence_dir=None):
    if one_sentence_dir is None:
        one_sentence_dir = ucca_parse_dir
    source, references, ranks = parse_grammatical_sentence(
        source, references, ranks, one_sentence_dir)
    return parse_Usim_sentence(source, references, ranks, ucca_parse_dir, filename)


def parse_combined_corpora(corpus_source, corpus_references, corpora, ucca_parse_dir, filename, one_sentence_dir=None):
    if one_sentence_dir is None:
        one_sentence_dir = ucca_parse_dir
    corpus_source, corpus_references, corpora = parse_grammatical_corpora(
        corpus_source, corpus_references, corpora, one_sentence_dir)
    return parse_Usim_corpora(corpus_source, corpus_references, corpora, ucca_parse_dir, filename)


def parse_Usim_sentence(source, references, ranks, ucca_parse_dir, filename):
    all_sentences = sentence_input_to_sentence_list(
        source, references, ranks)
    all_sentences = list(set(all_sentences))
    # with open(filename, "w") as fl:
    #     fl.write("\n".join(all_sentences))
    ucca_parse_sentences(all_sentences, ucca_parse_dir)
    return source, references, ranks


def parse_Usim_corpora(corpus_source, corpus_references, corpora, ucca_parse_dir, filename):
    all_sentences = corpus_input_to_sentence_list(
        corpus_source, corpus_references, corpora)
    all_sentences = list(set(all_sentences))
    # with open(filename, "w") as fl:
    # fl.write("\n".join(all_sentences))
    ucca_parse_sentences(all_sentences, ucca_parse_dir)
    return corpus_source, corpus_references, corpora


def parse_grammatical_sentence(source, references, ranks, one_sentence_dir):
    all_sentences = sentence_input_to_sentence_list(
        source, references, ranks)
    all_sentences = list(set(all_sentences))
    create_one_sentence_files(all_sentences, one_sentence_dir)
    return source, references, ranks


def parse_grammatical_corpora(corpus_source, corpus_references, corpora, one_sentence_dir):
    all_sentences = corpus_input_to_sentence_list(
        corpus_source, corpus_references, corpora)
    all_sentences = list(set(all_sentences))
    create_one_sentence_files(all_sentences, one_sentence_dir)
    return corpus_source, corpus_references, corpora


def main(args, parser):

    # initialize local arguments (Note globals are initiated at different
    # scope)
    seed = args.random_seed
    corpus_source_symb = args.corpus_source
    max_permutations = args.max_permutations
    debug = args.debug
    manual_analysis_num = args.manual_analysis
    matches_num = args.matches_num
    force = args.force
    if force:
        print("Forcing recalculation")

    # initialize version control parameters
    random.seed(seed)
    np.random.seed(seed)
    random_version = "" if corpus_source_symb != RAND else "seed" + str(
        seed) + "_"
    version = corpus_source_symb + "_" + \
        str(max_permutations) + "_" + random_version
    filename = version + "rank"

    # initialize cache dirs
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    sentence_cache_dir = os.path.join(CACHE_DIR, "sentence")
    if not os.path.isdir(sentence_cache_dir):
        os.makedirs(sentence_cache_dir)
    corpus_cache_dir = os.path.join(CACHE_DIR, "corpus")
    if not os.path.isdir(corpus_cache_dir):
        os.makedirs(corpus_cache_dir)
    parse_dir = os.path.join(CACHE_DIR, version + "parse")
    if not os.path.isdir(parse_dir):
        os.makedirs(parse_dir)
    one_sentence_dir = os.path.join(CACHE_DIR, "one_sentence")
    if not os.path.isdir(one_sentence_dir):
        os.makedirs(one_sentence_dir)

    # initialize references
    bn = os.path.join(REFERENCE_DIR, "BN")
    bn_refs = [bn + str(i) for i in [1, 2, 4, 5, 6, 8, 9, 10]]
    nucleA_ref = bn + "7"
    nucleB_ref = bn + "3"
    nucle_refs = [nucleA_ref, nucleB_ref]
    if ANNOTATION_FILE == CONLL_ANNOTATION_FILE:
        reference_files = bn_refs
        xml_ref_files = os.path.splitext(BN_ONLY_ANNOTATION_FILE)[0] + ".sgml"

    # sentence level inits
    ids_filename = os.path.join(sentence_cache_dir,  "id" + filename + ".json")
    ranks_filename = os.path.join(
        sentence_cache_dir,  "rank" + filename + ".json")
    sentence_parse_file = os.path.join(
        CACHE_DIR,  "sentence" + filename + ".txt")
    ranks, ids = create_ranks(ANNOTATION_FILE, max_permutations,
                              ranks_out_file=ranks_filename, ids_out_file=ids_filename)
    cache_scores = os.path.join(
        sentence_cache_dir,  "score" + filename + ".json")

    # corpus level inits
    corpora_ids_filename = os.path.join(
        corpus_cache_dir,  "corora_id" + filename + ".json")
    corpora_basename = os.path.join(
        corpus_cache_dir,  "corpus" + filename + ".json")
    # create corpus creating functions
    corpus_mean_corrections = [0, 2, 4, 6, 8, 10]
    exact_prob_vars = corpus_mean_corrections
    bin_prob_vars = [binomial_parameters_by_mean_and_var(
        i, 0.9) for i in corpus_mean_corrections]
    # create corpora
    prob_vars = bin_prob_vars
    # prob_vars = exact_prob_vars
    if prob_vars == bin_prob_vars:
        corpora_names = [str(x) + "," + str(y) for x, y in prob_vars]
    elif prob_vars == exact_prob_vars:
        corpora_names = [str(x) for x in prob_vars]
    corpora, corpus_ids = create_corpora(ANNOTATION_FILE, prob_vars,
                                         corpora_basename=corpora_basename, ids_out_file=corpora_ids_filename)
    corpus_parse_file = os.path.join(CACHE_DIR,  "corpus" + filename + ".txt")
    # create corpora source chooser
    if corpus_source_symb == SOURCE:
        choose_corpus_source = lambda x: x[0]
    elif corpus_source_symb == REF:
        choose_corpus_source = lambda x: x[-1]
    elif corpus_source_symb == RAND:
        choose_corpus_source = shuffle_sources
    else:
        print("unknown corpus source type")
        parser.print_help()
        parser.exit()

    print(
        "run all measures (Including combined 0,0.1...1 CombinedReference_less_callabale")
    measures = []
    add_measure(measures, r"iBLEU_{\alpha=0.8}", _ibleu_wrapper)
    gamma = 0.1
    add_measure(measures, "Reference_less_{gamma=" + str(gamma) + "}", CombinedReference_less_callabale(
        parse_dir, gamma), preprocess_sentence_level=lambda x, y, z: parse_combined_sentence(x, y, z, parse_dir, sentence_parse_file, parse_dir), preprocess_corpus_level=lambda x, y, z: parse_combined_corpora(x, y, z, parse_dir, sentence_parse_file))
    add_measure(measures, "USim", USim_callabale(
        parse_dir), preprocess_sentence_level=lambda x, y, z: parse_Usim_sentence(x, y, z, parse_dir, sentence_parse_file), preprocess_corpus_level=lambda x, y, z: parse_Usim_corpora(x, y, z, parse_dir, corpus_parse_file))
    add_measure(measures, "grammaticality", Grammaticallity_callabale(
        parse_dir), preprocess_sentence_level=lambda x, y, z: parse_grammatical_sentence(x, y, z, parse_dir), preprocess_corpus_level=lambda x, y, z: parse_grammatical_corpora(x, y, z, parse_dir))

    add_measure(measures, "BLEU", _bleu_wrapper)
    add_measure(measures, "M^2", _m2_wrapper, None, None, True)
    add_measure(measures, r"LD_{S\rightarrow O}", _levenshtein_wrapper)
    add_measure(measures, r"MinLD_{O\rightarrow R}",
                _levenshtein_references_wrapper)
    add_measure(measures, "GLEU", None, _glue_wrapper, None)
    add_measure(measures, "MAX_SARI", SARI_max_score)
    add_measure(measures, "SARI", SARI_score)
    mixmax = 100
    add_measure(measures, r"I-measure_{" + str(mixmax) + "mixmax}", None, Imeasure_callable(xml_ref_files, mixmax=mixmax),
                Imeasure_callable(xml_ref_files, return_per_sent_scores=False))
    mixmax = 1
    add_measure(measures, r"I-measure_{" + str(mixmax) + "mixmax}", None, Imeasure_callable(xml_ref_files, mixmax=mixmax),
                Imeasure_callable(xml_ref_files, return_per_sent_scores=False))
    add_measure(measures, r"I-measure_nomix", None, Imeasure_callable(xml_ref_files, mix=False),
                Imeasure_callable(xml_ref_files, return_per_sent_scores=False))

    assess_measures(measures, ranks, ids, corpora, corpus_ids,
                    reference_files, choose_corpus_source, corpora_names, corpus_mean_corrections, manual_analysis_num, matches_num, cache_scores, force)


def clean_tmp():
    if DEBUG:
        print("cleaning temporary cache files from", CACHE_DIR)
        assert "tmp" in CACHE_DIR
        if os.path.isdir(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)

if __name__ == '__main__':
    # Define and parse program input
    parser = argparse.ArgumentParser(description="Evaluate evaluation for GEC.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-ma", "--manual_analysis",
                        help="Maximum number of scores to print for manual analysis.", type=int, default=7)
    parser.add_argument("-f", "--force",
                        help="Force recalculation, ignore caches even if exists.", action="store_true")
    parser.add_argument("-p", "--max_permutations",
                        help="Maximum permutations of corrections appliance order per source,annotator pair.", type=int, default=1)
    parser.add_argument("-cs", "--corpus_source", help="Which corpus to choose as the source corpus.\n" +
                                                       SOURCE + " - NUCLE source corpus \n" +
                        REF + " - reference sentences\n" + RAND + " - a random choice of a source for each sentence", default=RAND)
    parser.add_argument("-s", "--random_seed",
                        help="random_seed.", type=int, default=1)
    parser.add_argument("-mn", "--matches_num",
                        help="Number of matches and mismatches of each measure with the human ranks to print.", type=int, default=0)
    parser.add_argument("-cache", "--cache_dir",
                        help="Directory to use for caching.", type=str, default=os.path.realpath(CACHE_DIR))
    parser.add_argument("-pool",
                        help="Pool size, default to the maximum detected.", type=int)
    parser.add_argument("-d", "--debug",
                        help="Debug mode doesn't use or save cache, also it runs only on few sentences.", action="store_true")

    args, _ = parser.parse_known_args()
    CACHE_DIR = args.cache_dir
    if args.pool is not None:
        POOL_SIZE = args.pool
    if args.debug:
        DEBUG = args.debug
        CACHE_DIR = os.path.join(CACHE_DIR, "tmp")
    try:
        main(args, parser)
    except:
        raise
    finally:
        clean_tmp()
