import numpy as np
import os
import sys
import re
import itertools
from utils import *
# ANNOTATION_FILE = BN_ANNOTATION_FILE
ANNOTATION_FILE = CONLL_ANNOTATION_FILE

NOOP = "noop"

def create_ranks(file, max_permutations=100000, filter_annot_chains=lambda x: True, min_annotators_per_sentence=0, ignore_noop=True, max_changes=None, ranks_out_file=None, ids_out_file=None):
    if ids_out_file is not None and ranks_out_file is not None:
        if os.path.isfile(ranks_out_file) and os.path.isfile(ids_out_file):
            print("reading ranks from file")
            return load_object_by_ext(ranks_out_file), load_object_by_ext(ids_out_file)
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

    total_sentences = 0
    total_annotations = 0
    print("annotators_num")
    ranks = []  # ranks[sentence][permutation][sentence, changes applied]
    sentence_ids = []
    for sentence_id, (source, all_changes) in enumerate(db):
        sentence_chains = []
        if min_annotators_per_sentence > len(all_changes):
            continue
        if ignore_noop:
            if find_in_iter(all_changes, NOOP):
                continue
        for annot_chains in all_changes:
            annot_chains = list(filter(filter_annot_chains, annot_chains))
            total_annotations += 1
            for permutation_id, changes in zip(range(max_permutations), itertools.permutations(annot_chains, max_changes)):
                rank = []
                for i in range(len(annot_chains) + 1):
                    rank.append(
                        (apply_changes(source, changes[:i]), changes[:i]))
                    total_sentences += 1
                    # if total_sentences % 1000 == 0:
                    #     print ("created a total of", total_sentences, "sentences")
                sentence_chains.append(rank)
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
            sentences = np.random.choice(sentence_chains)
            corrections_num = min(i, len(sentences) - 1)
            line = sentences[corrections_num][0]
            file.append(line)
        files.append(file)
    return files

# def combine_bn_with_alt(bn, alt, out)


def main():
    # combine_bn_with_alt(BN_ANNOTATION_FILE, CONLL_ANNOTATION_FILE, ANNOTATION_FILE)
    max_permutations = 1
    if ANNOTATION_FILE == BN_ANNOTATION_FILE:
        min_annotators_per_sentence = 10
        annot = "BN"
    elif ANNOTATION_FILE == CONLL_ANNOTATION_FILE:
        min_annotators_per_sentence = 2
        annot = "NUCLE"
    filename = str(max_permutations) + "_" + \
        str(min_annotators_per_sentence) + "_" + annot + "rank" + ".json"
    ids_filename = os.path.join(CACHE_DIR,  "id" + filename)
    ranks_filename = os.path.join(CACHE_DIR,  "id" + filename)
    ranks = create_ranks(ANNOTATION_FILE, max_permutations, ranks_out_file=ranks_filename,
                         ids_out_file=ids_filename, min_annotators_per_sentence=min_annotators_per_sentence)
    # print(ranks[0][:2])
    # print([x[:2] for x in create_levelled_files(ranks, 5)])


if __name__ == '__main__':
    main()
