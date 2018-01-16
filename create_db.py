import numpy as np
import os
import sys
import re
import itertools
import random
from utils import *
# ANNOTATION_FILE = BN_ANNOTATION_FILE
# print("not using alternatives")

ANNOTATION_FILE = CONLL_ANNOTATION_FILE

NOOP = "noop"


def parse_m2_to_db(file):
    db = []  # source, changes_per_annotator
    changes = []
    source = None
    with open(file, "r") as fl:
        for line in fl:
            line = line.strip()
            if line == "":
                assert source is not None
                db.append((source, split_changes_by_annot(changes)))
                changes = []
                source = None
            elif line.startswith("S"):
                source = line[2:]
            elif line.startswith("A"):
                properties = re.split("\|\|\|", line[2:])
                start, end = properties[0].split()
                properties[0] = end
                properties.insert(0, start)
                changes.append(properties)
            else:
                raise "unrecognized line " + line
    return db


def create_ranks(m2file, max_permutations=100000, filter_annot_changes=lambda x: True, min_annotators_per_sentence=0, ignore_noop=True, max_changes=None, ranks_out_file=None, ids_out_file=None, force=False):
    if not force and (ids_out_file is not None and ranks_out_file is not None):
        if os.path.isfile(ranks_out_file) and os.path.isfile(ids_out_file):
            print("reading ranks from file")
            return load_object_by_ext(ranks_out_file), load_object_by_ext(ids_out_file)
    db = parse_m2_to_db(m2file)
    total_sentences = 0
    total_annotations = 0
    ranks = []  # ranks[sentence][permutation][sentence, changes applied]
    sentence_ids = []
    for sentence_id, (source, all_changes) in enumerate(db):
        sentence_chains = []
        if min_annotators_per_sentence > len(all_changes):
            continue
        if ignore_noop:
            if find_in_iter(all_changes, NOOP):
                continue
        for annot_changes in all_changes:
            annot_changes = list(filter(filter_annot_changes, annot_changes))
            total_annotations += 1
            if max_changes is not None:
                cur_changes = min(max_changes, len(annot_changes))
            else:
                cur_changes = len(annot_changes)
            permutations_num = int(npermutations(list_to_hashable(annot_changes[:cur_changes])) * ncr(len(annot_changes), cur_changes))
            # if duplicate changes are possible in some future version use the following line instead of the one after (error might occur in subsetting annot_changes[:cur_changes])
            # if permutations_num < max_permutations and sum((1 for i, perm in zip(range(max_permutations + 1), itertools.permutations(annot_changes, max_changes)))) > max_permutations:
            if permutations_num < max_permutations:
                gen = itertools.permutations(annot_changes, max_changes)
            else:
                gen = (random.sample(annot_changes, cur_changes) for i in range(max_permutations)) # there exists a small chance of repeating chains
            for changes in gen:
                rank = []
                for i in range(len(annot_changes) + 1):
                    rank.append(
                        (apply_changes(source, changes[:i]), changes[:i]))
                    total_sentences += 1
                    # if total_sentences % 1000 == 0:
                    #     print ("created a total of", total_sentences, "sentences")
                sentence_chains.append(rank)
            # print("chain")
        if len(sentence_chains) > 1:
            sentence_ids.append(sentence_id)
            ranks.append(sentence_chains)
            if len(ranks) % 10 == 0:
                print("calculated for", len(ranks), "source sentences")
    print("Created", total_sentences, "sentences based on", len(ranks),
          "eligible sentences and a total of", total_annotations, "annotations.")
    if ids_out_file is not None:
        save_object_by_ext(sentence_ids, ids_out_file)
    if ranks_out_file is not None:
        save_object_by_ext(ranks, ranks_out_file)
    return ranks, sentence_ids


def create_levelled_files(ranks, file_num):
    """ creates parallel files by the order of the NUCLE sentences, choosing annotators randomly"""
    print("function was not checked")
    files = []
    for i in range(file_num):
        file = []
        for sentence_chains in ranks:
            sentences = choose_uniformely(sentence_chains)
            corrections_num = min(i, len(sentences) - 1)
            line = sentences[corrections_num][0]
            file.append(line)
        files.append(file)
    return files


def create_corpora(m2file, prob_vars, prob=None, num_sampled=1, filter_annot_changes=lambda x: True, min_annotators_per_sentence=0, ignore_noop=True, max_changes=None, corpora_basename=None, ids_out_file=None, force=False):

    prob_vars = np.array(prob_vars)
    if len(prob_vars.shape) == 1:
        prob_vars = np.expand_dims(prob_vars, axis=1)
    if prob is None:
        if prob_vars.shape[1] == 1:
            print("probability not specified, using prob_vars without variance")
            prob = lambda x: x
        elif prob_vars.shape[1] == 2:
            print("probability not specified, using binomial distribution")
            prob = np.random.binomial

    filenames = []
    if not force and (ids_out_file is not None and corpora_basename is not None):
        root = os.path.dirname(corpora_basename)
        basename = os.path.basename(corpora_basename)
        for vrs in prob_vars:
            repr_vars = ",".join([str(var) for var in vrs]) + "_"
            filename = os.path.join(root, repr_vars + basename)
            filenames.append(filename)
        if os.path.isfile(corpora_basename) and os.path.isfile(ids_out_file):
            print("reading corpora from file")
            corpora = [load_object_by_ext(filename) for filename in filenames]
            return corpora, load_object_by_ext(ids_out_file)
    db = parse_m2_to_db(m2file)
    corpora = []
    for vrs in prob_vars:
        sentence_ids = []
        sentences = []
        corpora.append(sentences)
        for sentence_id, (source, all_changes) in enumerate(db):
            if min_annotators_per_sentence > len(all_changes):
                continue
            if ignore_noop:
                if find_in_iter(all_changes, NOOP):
                    continue
            for i in range(num_sampled):
                all_changes = [list(filter(filter_annot_changes, annot_changes))
                               for annot_changes in all_changes]
                all_changes = [x for x in all_changes if x != []]
                if all_changes == []:
                    break
                changes = choose_uniformely(all_changes)
                # print(changes)
                changes = np.random.permutation(changes).tolist()
                # print(changes)
                changes_num = max(int(prob(*vrs)), 0)
                changes = changes[:changes_num]
                sentences.append((apply_changes(source, changes), changes))
                sentence_ids.append(sentence_id)
    for filename, corpus in zip(filenames, corpora):
        assert(len(sentence_ids) == len(corpus))
        save_object_by_ext(corpus, filename)
    if ids_out_file is not None:
        save_object_by_ext(sentence_ids, ids_out_file)
    return corpora, sentence_ids


def main():
    # combine_bn_with_alt(BN_ANNOTATION_FILE, CONLL_ANNOTATION_FILE, ANNOTATION_FILE)
    force = True
    max_permutations = 11
    if ANNOTATION_FILE == BN_ANNOTATION_FILE:
        min_annotators_per_sentence = 10
        annot = "BN"
    elif ANNOTATION_FILE == CONLL_ANNOTATION_FILE:
        min_annotators_per_sentence = 2
        annot = "NUCLE"
    filename = str(max_permutations) + "_" + \
        str(min_annotators_per_sentence) + "_" + annot + "rank" + ".json"
    ids_filename = os.path.join(CACHE_DIR,  "id" + filename)
    ranks_filename = os.path.join(CACHE_DIR,  "rank" + filename)
    corpora_ids_filename = os.path.join(CACHE_DIR,  "corora_id" + filename)
    corpora_basename = os.path.join(CACHE_DIR,  "corpus" + filename)

    ranks, ids = create_ranks(ANNOTATION_FILE, max_permutations, ranks_out_file=ranks_filename,
                              ids_out_file=ids_filename, min_annotators_per_sentence=min_annotators_per_sentence, force=force)
    corpus_sizes = [0, 2, 4, 6, 8]
    exact_prob_vars = corpus_sizes
    bin_prob_vars = [binomial_parameters_by_mean_and_var(
        i, 0.9) for i in corpus_sizes]
    prob_vars = bin_prob_vars
    # prob_vars = exact_prob_vars
    corpora, ids = create_corpora(ANNOTATION_FILE, prob_vars, min_annotators_per_sentence=min_annotators_per_sentence,
                                  corpora_basename=corpora_basename, ids_out_file=corpora_ids_filename, force=force)
    # print(corpora[:2])
    print("wrong number of corrections")
    print([corpus[:2] for corpus in corpora])
    # print(ranks[0][:2])
    # print([x[:2] for x in create_levelled_files(ranks, 5)])


if __name__ == '__main__':
    main()
