import os
import re
import numpy as np
import json
import pickle

PROJECT = os.path.realpath(os.path.dirname(__file__)) + os.sep
CACHE_DIR = os.path.join(PROJECT, "cache")
if not os.path.isdir(CACHE_DIR):
    os.makedirs(CACHE_DIR)

ASSESS_DIR = os.path.join(PROJECT, "assess_learner_language")
DATA_DIR = os.path.join(ASSESS_DIR, "data")
REFERENCE_DIR = os.path.join(DATA_DIR, "references")

CONLL_ANNOTATION_FILE = os.path.join(
    PROJECT, "conll14st-test-data", "alt", "official-2014.combined-withalt.m2")
print("not using alternatives")
BN_ANNOTATION_FILE = os.path.join(REFERENCE_DIR, "BN_corrected.m2")


def load_object_by_ext(filename):
    ext = os.path.splitext(filename)[1]
    if ext == ".json":
        with open(filename, "r") as fl:
            return json.load(fl)
    elif ext in [".pkl", ".pckl", ".pickl", ".pickle"]:
        with open(filename, "rb") as fl:
            return pickle.load(fl)
    else:
        raise "format not supported" + ext


def save_object_by_ext(obj, filename):
    ext = os.path.splitext(filename)[1]
    if ext == ".json":
        with open(filename, "w") as fl:
            return json.dump(obj, fl)
    elif ext in [".pkl", ".pckl", ".pickl", ".pickle"]:
        with open(filename, "wb") as fl:
            return pickle.dump(obj, fl)
    else:
        raise "format not supported" + ext


def get_lines_from_file(filename, lines, normalize=lambda x: x):
    with open(filename) as fl:
        text = np.array(fl.readlines())
        if lines is not None:
            lines = np.array(lines)
            text = text[lines]
        return (normalize(line.replace("\n", "")) for line in text)


def apply_changes(sentence, changes):
    changes = sorted(changes, key=lambda x: (int(x[0]), int(x[1])))
    res = []
    last_end = 0
    s = sentence.split()
    for change in changes:
        start = int(change[0])
        assert last_end == 0 or last_end <= start, "changes collide in places:" + \
            str(last_end) + ", " + str(start) + \
            "\nSentence: " + sentence + "\nChanges " + str(changes)
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


def iterate_chains(ranks, ids=None):
    if ids == None:
        ids = itertools.repeat(ids)
    else:
        assert len(ids) == len(ranks)

    for sentence_chains, sentence_id in zip(ranks, ids):
        for chain in sentence_chains:
            if sentence_id is not None:
                yield chain, sentence_id
            else:
                yield chain


def traverse_ranks(ranks):
    for tple in traverse_chains(iterate_chains(ranks)):
        yield tple


def traverse_chains(chains):
    for chain in chains:
        for tple in chain:
            yield tple
