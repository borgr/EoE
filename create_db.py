import numpy as np
import os
import sys
import re
import itertools
import json
import pickle
from utils import *

PROJECT = os.path.realpath(os.path.dirname(__file__)) + os.sep
CACHE_DIR = os.path.join(PROJECT, "cache")
if not os.path.isdir(CACHE_DIR):
    os.makedirs(CACHE_DIR)
ANNOTATION_FILE = os.path.join(
    PROJECT, "conll14st-test-data", "alt", "official-2014.combined-withalt.m2")
NOOP = "noop"

def iterate_chains(ranks, ids=None):
    if ids == None:
        ids = itertools.repeat(ids)
    else:
        assert len(ids) == len(ranks)

    for sentence_id, sentence_chains in zip(ranks, ids):
        for chain in sentence:
            if sentence_id is not None:
                yield chain, sentence_id
            else:
                yield chain

def iterate_sentence_changes(ranks):
    for chain in iterate_chains(ranks):
        for tple in chain:
            yield tple

def apply_changes(sentence, changes):
    changes = sorted(changes, key=lambda x: (int(x[0]), int(x[1])))
    res = []
    last_end = 0
    s = sentence.split()
    for change in changes:
        start = int(change[0])
        assert last_end == 0 or last_end <= start, "changes collide in places:" + \
            str(last_end) + "," + str(start) + str(changes)
        if start == -1:
            print("noop action, no change applied")
            assert change[2] == NOOP
            print(changes)
            raise changes
            return sentence
        res += s[last_end:start] + [change[3]]
        last_end = int(change[1])
    res += s[last_end:]
    return re.sub("\s+", " ", " ".join(res))


def split_changes_by_annot(changes):
    res = {}
    for change in changes:
        annot = change[-1]
        if annot not in res:
            res[annot] = []
        res[annot].append(change)
    return list(res.values())


def find_in_iter(iterable, key):
    if hasattr(iterable, '__iter__') and type(iterable) != type(key):
        return any((find_in_iter(item, key) for item in iterable))
    return key == iterable


def create_ranks(file, max_permutations=100, filter_annot_chains=lambda x: True, ignore_noop=True, max_changes=None, ranks_out_file=None, ids_out_file=None):
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
    ranks = []  # ranks[sentence][permutation][sentence, changes applied]
    sentence_ids = []
    for sentence_id, (source, all_changes) in enumerate(db):
        sentence_chains = []
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
            sentences = sentence_chains[
                np.random.randint(len(sentence_chains))]
            corrections_num = min(i, len(sentences) - 1)
            line = sentences[corrections_num][0]
            file.append(line)
        files.append(file)
    return files


def main():
    max_permutations = 1
    filename = str(max_permutations) + "_" + "rank" + ".json"
    ids_filename = os.path.join(CACHE_DIR,  "id" + filename)
    ranks_filename = os.path.join(CACHE_DIR,  "id" + filename)
    ranks = create_ranks(ANNOTATION_FILE, max_permutations, ranks_out_file=ranks_filename, ids_out_file=ids_filename)
    print(ranks[0][:2])
    # print([x[:2] for x in create_levelled_files(ranks, 5)])


if __name__ == '__main__':
    main()
