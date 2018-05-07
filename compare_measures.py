import os
import sys
import shutil
from itertools import repeat
import multiprocessing
import subprocess
import shlex
import argparse
import time
from datetime import datetime
import random
from threading import Lock
import re

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

import imeasure.ieval as iev

UCCA_MODEL_PATH = "/hard/coded/path/models/bilstm"

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
MULTIPROCESSED = "multiprocessed"

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
def convert2edits(sources, all_references, cache_file=None, return_filename=False):
    to_hash = (tuple(list_to_hashable(sources, True)), tuple(
        list_to_hashable(all_references, True)))
    hash_key = get_hash(to_hash)
    if cache_file is None:
        cache_file = os.path.join(
            CACHE_DIR, 'edits_' + hash_key + '.m2')
    if hash_key in _convert2edits_cache:
        return _convert2edits_cache[hash_key]
    if os.path.isfile(cache_file):
        return load_object_by_ext(cache_file)
    res = p2m2.parallel_to_m2(sources, all_references)
    res = [r + "\n" if r.startswith("S") else r for r in res]
    save_object_by_ext(res, cache_file)
    _, res = m2scorer.load_annotation(cache_file)
    _convert2edits_cache[hash_key] = res
    if return_filename:
        return cache_file
    return res


def score_corpus(sources, all_references, corpus, sentence_measure, corpus_measure=None, edit_based=False, cache_file=None, force=False):
    if not force and cache_file is not None:
        if os.path.isfile(cache_file):
            # print("reading measure from cache", cache_file)
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


def score_corpora(sources, all_references, corpora, sentence_measure, corpus_measure=None, edit_based=False, allow_multiprocessing=True, cache_files=None, force=False):
    print("score corpora")
    if cache_files is None:
        cache_files = [None] * len(corpora)
    if allow_multiprocessing:
        pool = multiprocessing.Pool(POOL_SIZE)
        scores = pool.starmap(score_corpus, zip(repeat(sources), repeat(all_references), corpora, repeat(sentence_measure),
                                                repeat(corpus_measure), repeat(edit_based), cache_files, repeat(force)))
        pool.close()
        pool.join()
    else:
        scores = [score_corpus(*args) for args in zip(repeat(sources), repeat(all_references), corpora, repeat(sentence_measure),
                                                      repeat(corpus_measure), repeat(edit_based), cache_files, repeat(force))]
    return scores


def measure_corpora(sources, all_references, corpora, sentence_measure, corpus_measure=None, edit_based=False, allow_multiprocessing=True, cache_files=None, corpora_scores=None, force=False):
    aggregated_corpora_scores = score_corpora(
        sources, all_references, corpora, None, corpus_measure, edit_based, allow_multiprocessing, cache_files, force)
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
            # print("reading measure from cache", cache_file)
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
        scores_nonit = corpus_measure(
            corpus_sources, corpus_references, sentences)
        try:
            scores_it = iter(scores_nonit)
        except TypeError:
            print(scores_nonit)
            raise ValueError(
                "corpus measure should be None or a function that returns an iterator or an iterable")
    try:
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
    except:
        print("scores from corpus measure:")
        print(scores_nonit)
        print("len", len(scores_nonit))
        print("supposed to be len", len(corpus_sources),
              len(corpus_references), len(sentences))
        raise


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


def assess_measures(measures, ranks, ids, corpora, corpus_ids, reference_files, choose_source=choose_source_per_chain, choose_corpus_source=lambda x: x[SENTENCE_LOC], corpora_names=None, corpora_scores=None, manual_analysis_num=True, vma=False, matches_num=0, cache=None, force=False):
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
    sources = choose_source(ranks)
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
        sources = sources[:4]
        references = references[:4]
        ranks = ranks[:4]
        corpus_source = corpus_source[:4]
        corpus_references = corpus_references[:4]
        corpora = [x[:4] for x in corpora]
    human_scores = ranks_to_scores(ranks)
    print("Overall corpus statistics:")
    print("Number of sentences in the corpus:",
          len(list(traverse_ranks(ranks))))
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
        allow_multiprocessing = not from_measure(
            measure_details, MULTIPROCESSED)
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
                p_corpus_source, p_corpus_references, p_corpora, sentence_measure, corpus_measure, edit_based, allow_multiprocessing, cache_files, force)
            aggregated_corpora_scores = np.mean(measure_corpora_scores, 1)
        else:
            aggregated_corpora_scores = measure_corpora(
                p_corpus_source, p_corpus_references, p_corpora, None, corpus_scorer, edit_based, allow_multiprocessing, cache_files, corpora_scores, force)
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
                              human_flatten_scores, manual_analysis_num, vma, sources, references, list(iterate_chains(ranks)))
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


def print_list_statistics(x, y, manual_analysis_num, vma=False, sources=None, all_references=None, chains=None):
    assert len(x) == len(y), " Lists were expected to be in the same length and not " + \
        str(len(y)) + ", " + str(len(x))
    pearson = pearsonr(y, x)
    print("Pearson (val, P-val):", pearson[0], pearson[1])
    spearman = spearmanr(y, x)
    print("Spearman (val, P-val):", spearman[0], spearman[1])
    # kendall = kendalltau(y, x)
    # print("Kendall's Tau (val, P-val):", kendall[0], kendall[1])
    if manual_analysis_num:
        if vma and sources is not None and all_references is not None and chains is not None:
            i = 0
            print("First sources, references, and sentences given to the measure")
            for source, references, chain in zip(sources, all_references, chains):
                print("source, references")
                print(source, references)
                print("sentences together with their corresponding ranks")
                for sentence in chain:
                    i += 1
                    if i == manual_analysis_num:
                        break
                    print(sentence, x[i], y[i])
                if i == manual_analysis_num:
                    break
        else:
            print("First", manual_analysis_num,
                  "scores for manual analysis, together with their corresponding ranks")
            print(np.array(list(zip(y, x)))[
                  :manual_analysis_num].transpose())


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
    if attribute == MULTIPROCESSED:
        return measure_details[7]


def add_measure(measures, name, sentence_measure=None, corpus_measure=None, corpus_scorer=None, edit_based=False, preprocess_sentence_level=lambda x, y, z: (x, y, z), preprocess_corpus_level=lambda x, y, z: (x, y, z), multiprocessed=False):
    """
    measures - a list with the measures used so far
    name - name of the added measure
    sentence_measure - a function to calculate the measure per source, references, system
    corpus_measure - a function to calculate an iterable of measures, one per each in the iterables sources, references_list, system_outputs
    corpus_scorer - a function that returns a single score for the iterables sources, references_list, system_outputs
    edit_based - whther the measure expects references to be edits instead of sentences
    preprocess_sentence_level - how to preprocess each sentence
    preprocess_corpus_level - how to preprocess the entire corpora (iterables sources, references_list, system_outputs)
    multiprocessed - whether the measure is already using multiprocessing (and hence e.g. processing different corpora scores in parallel is prohibited)
    """
    assert not (
        sentence_measure is None and corpus_measure is None and corpus_scorer is None)
    measures.append((name, sentence_measure, corpus_measure,
                     corpus_scorer, edit_based, preprocess_sentence_level, preprocess_corpus_level, multiprocessed))


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
    ucca_parse_sentences(all_sentences, ucca_parse_dir, UCCA_MODEL_PATH)
    return source, references, ranks


def parse_Usim_corpora(corpus_source, corpus_references, corpora, ucca_parse_dir, filename):
    all_sentences = corpus_input_to_sentence_list(
        corpus_source, corpus_references, corpora)
    all_sentences = list(set(all_sentences))
    ucca_parse_sentences(all_sentences, ucca_parse_dir, UCCA_MODEL_PATH)
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
    source_symb = args.source
    print("Source sentences are chosen to be " + source_symb + " sentences")
    max_permutations = args.max_permutations
    debug = args.debug
    manual_analysis_num = args.manual_analysis
    vma = args.verbose_manual_analysis
    matches_num = args.matches_num
    force = args.force
    if force:
        print("Forcing recalculation")

    # initialize version control parameters
    random.seed(seed)
    np.random.seed(seed)
    random_version = "" if source_symb != RAND else "seed" + str(
        seed) + "_"
    version = source_symb + "_" + \
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
    corpus_mean_corrections = np.linspace(0, 10, 11)
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
    if source_symb == SOURCE:
        print("Using origin as source")
        choose = lambda x: x[0]
        choose_corpus_source = choose
        choose_sentence_source = lambda x: choose_source_per_chain(x, choose)
    elif source_symb == REF:
        print("Using reference as source")
        choose = lambda x: x[-1]
        choose_corpus_source = choose
        choose_sentence_source = lambda x: choose_source_per_chain(x, choose)
    elif source_symb == RAND:
        print("Using random source")
        choose_corpus_source = shuffle_sources
        choose_sentence_source = choose_source_per_chain
    else:
        print("unknown corpus source type")
        parser.print_help()
        parser.exit()

    measures = []
    #####################################################################################################
    ####      add a call for each measure of choice
    #####################################################################################################

    # if source_symb == SOURCE and ANNOTATION_FILE == CONLL_ANNOTATION_FILE:
    # add_measure(measures, r"imeasure ref num", None, Imeasure_num_callable(),
    # Imeasure_num_callable(return_per_sent_scores=False),
    # multiprocessed=True)

    # add_measure(measures, r"I-measure_nomix", None, Imeasure_callable(mix=False),
    #             Imeasure_callable(return_per_sent_scores=False, mix=False), multiprocessed=True)

    add_measure(measures, "USim", USim_callabale(
        parse_dir), preprocess_sentence_level=lambda x, y, z: parse_Usim_sentence(x, y, z, parse_dir, sentence_parse_file), preprocess_corpus_level=lambda x, y, z: parse_Usim_corpora(x, y, z, parse_dir, corpus_parse_file))
    # add_measure(measures, r"iBLEU_{\alpha=0.8}", _ibleu_wrapper)
    # add_measure(measures, "M^2", _m2_wrapper, None, None, True)
    # gamma = 0.1
    # add_measure(measures, "Reference_less_{gamma=" + str(gamma) + "}", CombinedReference_less_callabale(
    #     parse_dir, gamma), preprocess_sentence_level=lambda x, y, z: parse_combined_sentence(x, y, z, parse_dir, sentence_parse_file, parse_dir), preprocess_corpus_level=lambda x, y, z: parse_combined_corpora(x, y, z, parse_dir, sentence_parse_file))
    # add_measure(measures, "Grammaticality", Grammaticallity_callabale(
    # parse_dir), preprocess_sentence_level=lambda x, y, z:
    # parse_grammatical_sentence(x, y, z, parse_dir),
    # preprocess_corpus_level=lambda x, y, z: parse_grammatical_corpora(x, y,
    # z, parse_dir))

    # add_measure(measures, "BLEU", _bleu_wrapper)
    # add_measure(measures, r"LD_{S\rightarrow O}", _levenshtein_wrapper)
    # add_measure(measures, r"MinLD_{O\rightarrow R}",
    #             _levenshtein_references_wrapper)
    # add_measure(measures, "GLEU", None, _glue_wrapper, None)
    # add_measure(measures, "MAX_SARI", SARI_max_score)
    # add_measure(measures, "SARI", SARI_score)
    # mixmax = 100
    # add_measure(measures, r"I-measure_{" + str(mixmax) + "mixmax}", None, Imeasure_callable(mixmax=mixmax),
    # Imeasure_callable(return_per_sent_scores=False, mixmax=mixmax))

    assess_measures(measures, ranks, ids, corpora, corpus_ids,
                    reference_files, choose_sentence_source, choose_corpus_source, corpora_names, corpus_mean_corrections, manual_analysis_num, vma, matches_num, cache_scores, force)


def clean_tmp():
    if DEBUG:
        print("cleaning temporary cache files from", CACHE_DIR)
        assert "tmp" in CACHE_DIR
        if os.path.isdir(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)

###############################################################################################
##### measures implementation examples
###############################################################################################

class Imeasure_num_callable(object):
    calls = 0
    ref_files = {}
    cache_dir = os.path.join(CACHE_DIR, "im")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    mutex = Lock()
    score_cache_file = os.path.join(cache_dir, "scores.pkl")
    if os.path.isfile(score_cache_file):
        score_cache = load_object_by_ext(score_cache_file)
    else:
        score_cache = {}

    def __init__(self, return_per_sent_scores=True, mixmax=100, cache_file=None, **kwargs):
        self.return_per_sent_scores = return_per_sent_scores
        self.mixmax = mixmax
        self.bound_args = kwargs

    def get_key(source, sentence):
        return (source, sentence)

    def add_to_cache(source, sentence, score):
        Imeasure_num_callable.mutex.acquire()
        Imeasure_num_callable.score_cache[
            Imeasure_num_callable.get_key(source, sentence)] = score
        if len(Imeasure_num_callable.score_cache) % 100 == 0:
            print("Imeasure cache size", len(
                Imeasure_num_callable.score_cache))
            Imeasure_num_callable.score_cache = load_object_by_ext(
                Imeasure_num_callable.score_cache_file)
        Imeasure_num_callable.mutex.release()

    def get_from_cache(source, sentence):
        return Imeasure_num_callable.score_cache[Imeasure_num_callable.get_key(source, sentence)]

    def in_cache(source, sentence):
        return Imeasure_num_callable.get_key(source, sentence) in Imeasure_num_callable.score_cache

    def split_xml(xml, write=True):
        begining = """<?xml version='1.0' encoding='UTF-8'?>\n<scripts><script id="1">"""
        ending = "</script></scripts>"
        with open(xml) as fl:
            content = fl.read()
        sentences = re.findall("<sentence.*?ntence>", content)
        contents = [begining + sentence + ending for sentence in sentences]
        if not write:
            return contents
        xmls = []
        xml_dir = os.path.splitext(xml)[0]
        if not os.path.isdir(xml_dir):
            try:
                os.makedirs(xml_dir)
            except FileExistsError:
                pass
        for i, content in enumerate(contents):
            xml_file = os.path.join(xml_dir, str(i) + os.path.splitext(xml)[1])
            xmls.append(xml_file)
            with open(xml_file, "w") as fl:
                fl.write(content)
        return xmls

    def get_xml(sources, references):
        if len(sources) != len(references):
            references = np.array(references).transpose()
        cache_file = convert2edits(sources, references, return_filename=True)

        to_hash = (tuple(list_to_hashable(sources, True)),
                   tuple(list_to_hashable(references, True)))
        hash_key = get_hash(to_hash)
        if hash_key in Imeasure_num_callable.ref_files:
            return Imeasure_num_callable.ref_files[hash_key]
        xml_file = os.path.join(
            Imeasure_num_callable.cache_dir, "im_" + hash_key + ".xml")
        if not os.path.isfile(xml_file):
            command = ASSESS_DIR + os.sep + "m2_to_ixml.sh -in:" + cache_file + \
                " -out:" + xml_file
            print("command", command)
            p = subprocess.Popen(shlex.split(command))
            p.communicate()
        Imeasure_num_callable.ref_files[hash_key] = xml_file
        return xml_file

    def Imeasure_num_scores(source, file_ref, system, **kwargs):
        """ If per_sentence_score is False, one accumulated score is returned instead of a score for each sentence"""
        file_hyp = None
        if isinstance(system, six.string_types):
            file_hyp = system
            system = None
        return iev.count_clusters(file_ref, file_hyp=file_hyp, hyps=system, **kwargs)

    def ieval(self, source, ref_file, sentence, mixmax, quiet=True, **kwargs):
        # We do not lock here, a sentence might sometimes be calculated twice,
        # in which case the last calculation will be cached
        if Imeasure_num_callable.in_cache(source, sentence):
            return Imeasure_num_callable.get_from_cache(source, sentence)
        res = Imeasure_num_callable.Imeasure_num_scores([source], ref_file, [sentence],
                                                        return_counts=True, mixmax=self.mixmax, quiet=quiet, **self.bound_args, **kwargs)
        return res

    def __call__(self, sources, references, sentences, **kwargs):
        files = Imeasure_num_callable.split_xml(
            Imeasure_num_callable.get_xml(sources, references))

        pool = multiprocessing.Pool(POOL_SIZE)
        scores = pool.starmap(self.ieval, zip(sources, files, sentences,
                                              repeat(self.mixmax)))
        pool.close()
        pool.join()
        print("scores", scores)
        scores = [score[0] for score in scores]
        print("Candidates nums", scores)
        return scores


class Imeasure_callable(object):
    calls = 0
    ref_files = {}
    cache_dir = os.path.join(CACHE_DIR, "im")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    mutex = Lock()
    score_cache_file = os.path.join(cache_dir, "scores.pkl")
    if os.path.isfile(score_cache_file):
        score_cache = load_object_by_ext(score_cache_file)
    else:
        score_cache = {}

    def __init__(self, return_per_sent_scores=True, mixmax=100, cache_file=None, **kwargs):
        self.return_per_sent_scores = return_per_sent_scores
        self.mixmax = mixmax
        self.bound_args = kwargs

    def get_key(source, sentence):
        return (source, sentence)

    def add_to_cache(source, sentence, score):
        Imeasure_callable.mutex.acquire()
        Imeasure_callable.score_cache[
            Imeasure_callable.get_key(source, sentence)] = score
        if len(Imeasure_callable.score_cache) % 100 == 0:
            print("Imeasure cache size", len(Imeasure_callable.score_cache))
            save_object_by_ext(Imeasure_callable.score_cache,
                               Imeasure_callable.score_cache_file)
        Imeasure_callable.mutex.release()

    def get_from_cache(source, sentence):
        return Imeasure_callable.score_cache[Imeasure_callable.get_key(source, sentence)]

    def in_cache(source, sentence):
        return Imeasure_callable.get_key(source, sentence) in Imeasure_callable.score_cache

    def split_xml(xml, write=True):
        begining = """<?xml version='1.0' encoding='UTF-8'?>\n<scripts><script id="1">"""
        ending = "</script></scripts>"
        with open(xml) as fl:
            content = fl.read()
        sentences = re.findall("<sentence.*?ntence>", content)
        contents = [begining + sentence + ending for sentence in sentences]
        if not write:
            return contents
        xmls = []
        xml_dir = os.path.splitext(xml)[0]
        if not os.path.isdir(xml_dir):
            try:
                os.makedirs(xml_dir)
            except FileExistsError:
                pass
        for i, content in enumerate(contents):
            xml_file = os.path.join(xml_dir, str(i) + os.path.splitext(xml)[1])
            xmls.append(xml_file)
            with open(xml_file, "w") as fl:
                fl.write(content)
        return xmls

    def get_xml(sources, references):
        if len(sources) != len(references):
            references = np.array(references).transpose()
        cache_file = convert2edits(sources, references, return_filename=True)

        to_hash = (tuple(list_to_hashable(sources, True)),
                   tuple(list_to_hashable(references, True)))
        hash_key = get_hash(to_hash)
        if hash_key in Imeasure_num_callable.ref_files:
            return Imeasure_num_callable.ref_files[hash_key]
        xml_file = os.path.join(
            Imeasure_num_callable.cache_dir, "im_" + hash_key + ".xml")
        if not os.path.isfile(xml_file):
            command = ASSESS_DIR + os.sep + "m2_to_ixml.sh -in:" + cache_file + \
                " -out:" + xml_file
            print("command", command)
            p = subprocess.Popen(shlex.split(command))
            p.communicate()
        Imeasure_num_callable.ref_files[hash_key] = xml_file
        return xml_file

    def get_score(self, imeasore_score):
        return imeasore_score["c"]["wacc"] * 100

    def ieval(self, source, ref_file, sentence, mixmax, quiet=True, **kwargs):
        # We do not lock here, a sentence might sometimes be calculated twice,
        # in which case the last calculation will be cached
        if Imeasure_callable.in_cache(source, sentence):
            return Imeasure_callable.get_from_cache(source, sentence)
        res = Imeasure_scores([source], ref_file, [sentence],
                              return_counts=True, mixmax=self.mixmax, quiet=quiet, **self.bound_args, **kwargs)
        Imeasure_callable.add_to_cache(source, sentence, res)
        return res

    def __call__(self, sources, references, sentences, **kwargs):
        files = Imeasure_callable.split_xml(
            Imeasure_callable.get_xml(sources, references))

        pool = multiprocessing.Pool(POOL_SIZE)
        scores = pool.starmap(self.ieval, zip(sources, files, sentences,
                                              repeat(self.mixmax)))
        pool.close()
        pool.join()
        if self.return_per_sent_scores:
            res = [iev.compute_all(sys, base) for sys, base in scores]
            res = [self.get_score(r) for r in res]
        else:
            print(scores)
            res = reduce(lambda x, y: (iev.add_counter_counter(
                x[0], y[0]), iev.add_counter_counter(x[1], y[1])), zip(*scores))
            assert len(res) == 2
            res = iev.compute_all(res[0], res[1])
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


if __name__ == '__main__':
    # Define and parse program input
    parser = argparse.ArgumentParser(description="Evaluate evaluation for GEC.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-ma", "--manual_analysis",
                        help="Maximum number of scores to print for manual analysis.", type=int, default=7)
    parser.add_argument("-vma", "--verbose_manual_analysis",
                        help="Print more for manual analysis.", action="store_true")
    parser.add_argument("-f", "--force",
                        help="Force recalculation, ignore caches even if exists.", action="store_true")
    parser.add_argument("-p", "--max_permutations",
                        help="Maximum permutations of corrections appliance order per source,annotator pair.", type=int, default=1)
    parser.add_argument("-s", "--source", help="Which sentences to choose as the source (in sentence chain or corpus id).\n" +
                        SOURCE + " - NUCLE source \n" +
                        REF + " - reference sentences\n" + RAND + " - a random choice of a source for each sentence", default=RAND)
    parser.add_argument("-rs", "--random_seed",
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
        print("pool size is set to ", POOL_SIZE)
    if args.debug:
        DEBUG = args.debug
        CACHE_DIR = os.path.join(CACHE_DIR, "tmp")
    try:
        main(args, parser)
    except:
        raise
    finally:
        clean_tmp()
