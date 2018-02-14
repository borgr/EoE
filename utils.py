import os
import itertools
import re
import numpy as np
import json
import pickle
import operator
import hashlib
from functools import reduce
from collections import Counter
from math import factorial
from scipy.stats import norm
import math
import six
from itertools import repeat

# from scipy.stats.stats import kendalltau
PROJECT = os.path.realpath(os.path.dirname(__file__)) + os.sep
CACHE_DIR = os.path.join(PROJECT, "cache")
if not os.path.isdir(CACHE_DIR):
    os.makedirs(CACHE_DIR)

ASSESS_DIR = os.path.join(PROJECT, "assess_learner_language")
DATA_DIR = os.path.join(ASSESS_DIR, "data")
REFERENCE_DIR = os.path.join(DATA_DIR, "references")

CONLL_ANNOTATION_FILE = os.path.join(
    PROJECT, "conll14st-test-data", "alt", "official-2014.combined-withalt.m2")

BN_ANNOTATION_FILE = os.path.join(REFERENCE_DIR, "BN_corrected.m2")
BN_ANNOTATION_FILE = os.path.join(REFERENCE_DIR, "BN_using_conll_script.m2")
BN_ONLY_ANNOTATION_FILE = os.path.join(REFERENCE_DIR, "BNonly.m2")


##########################################################################
# File Manipulation
##########################################################################

def load_object_by_ext(filename):
    ext = os.path.splitext(filename)[1]
    if ext == ".json":
        with open(filename, "r") as fl:
            return json.load(fl)
    elif ext in [".pkl", ".pckl", ".pickl", ".pickle"]:
        with open(filename, "rb") as fl:
            return pickle.load(fl)
    elif ext in [".txt", ".log", ".out", ".crps", ".m2", ".sgml"]:
        with open(filename, "r") as fl:
            return fl.read()
    else:
        raise Exception("format not supported" + ext)


def save_object_by_ext(obj, filename):
    ext = os.path.splitext(filename)[1]
    if ext == ".json":
        with open(filename, "w") as fl:
            return json.dump(obj, fl)
    elif ext in [".pkl", ".pckl", ".pickl", ".pickle"]:
        with open(filename, "wb") as fl:
            return pickle.dump(obj, fl)
    elif ext in [".txt", ".log", ".out", ".crps", ".m2", ".sgml"]:
        with open(filename, "w") as fl:
            if isinstance(obj, six.string_types):
                return fl.write(obj)
            else:
                return fl.writelines(obj)
    else:
        raise Exception("format not supported" + ext)


def get_lines_from_file(filename, lines, normalize=lambda x: x):
    with open(filename) as fl:
        text = np.array(fl.readlines())
        if lines is not None:
            lines = np.array(lines)
            text = text[lines]
        return (normalize(line.replace("\n", "")) for line in text)

##########################################################################
# General
##########################################################################

list_types = [type([]), type(np.array([]))]
def list_to_hashable(lst, sorting=False):
    if type(lst) not in list_types:
        return lst
    if sorting:
        lst.sort()
    return tuple((list_to_hashable(x) for x in lst))

def get_hash(hashable):
    hasher = hashlib.md5()
    hasher.update(str(hashable).encode("utf-8"))
    print("hash", hasher.hexdigest())
    return hasher.hexdigest()


def choose_uniformely(lst):
    return lst[np.random.randint(len(lst))]


def binomial_parameters_by_mean_and_var(mean, var):
    if mean == 0:
        return 1, 0
    p = (mean - var) / mean
    n = mean / p  # explicitly: (mean ** 2) / (mean - var)
    return n, p


def npermutations(l):
    num = factorial(len(l))
    mults = Counter(l).values()
    den = reduce(operator.mul, (factorial(v) for v in mults), 1)
    return num / den


def ncr(n, r):
    r = min(r, n - r)
    if r == 0:
        return 1
    numer = reduce(operator.mul, range(n, n - r, -1))
    denom = reduce(operator.mul, range(1, r + 1))
    return numer // denom


def kendall_mergesort(offs, length, x, y, perm, temp):
    exchcnt = 0
    if length == 1:
        return 0
    if length == 2:
        if y[perm[offs]] <= y[perm[offs + 1]]:
            return 0
        t = perm[offs]
        perm[offs] = perm[offs + 1]
        perm[offs + 1] = t
        return 1
    length0 = length // 2
    length1 = length - length0
    middle = offs + length0
    exchcnt += kendall_mergesort(offs, length0, x, y, perm, temp)
    exchcnt += kendall_mergesort(middle, length1, x, y, perm, temp)
    if y[perm[middle - 1]] < y[perm[middle]]:
        return exchcnt
    # merging
    i = j = k = 0
    while j < length0 or k < length1:
        if k >= length1 or (j < length0 and y[perm[offs + j]] <= y[perm[middle + k]]):
            temp[i] = perm[offs + j]
            d = i - j
            j += 1
        else:
            temp[i] = perm[middle + k]
            d = (offs + i) - (middle + k)
            k += 1
        if d > 0:
            exchcnt += d
        i += 1
    perm[offs:offs + length] = temp[0:length]
    return exchcnt


def kendall_partial_order_from_seq(xs, ys, sentences, ids=None):
    """ calculates tau-a, assumes no ties"""
    nd = 0
    # currently assumes list of ranks instead of many lists which may contain repetitions should use x_ids, y_ids to remove counting pairs that were seen already
    exchanges = []
    pairs = set()
    sequence_of_lists = True
    if ids is None:
        ids = repeat(repeat(None))
        sequence_of_lists = True
    for sub_x, sub_y, sub_ids in zip(xs, ys, ids):
        last = None
        for i, (x_i, y_i, i_id) in enumerate(zip(sub_x, sub_y, sub_ids)):
            half_sub_ids = sub_ids[i + 1:] if not sequence_of_lists else repeat(None)
            for x_j, y_j, j_id in zip(sub_x[i + 1:], sub_y[i + 1:], half_sub_ids):
                if sequence_of_lists or (i_id, j_id) not in pairs:
                    if sequence_of_lists:
                        pairs.add(len(pairs))
                    else:
                        pairs.add((i_id, j_id))
                    x_dir = x_i - x_j > 0
                    assert x_i != x_j
                    y_dir = y_i - y_j > 0
                    if x_dir != y_dir and y_i != y_j:
                        nd += 1
    pairs_num_sqrt = math.sqrt(len(pairs))
    z = 2 * nd / pairs_num_sqrt - pairs_num_sqrt
    p = 2 * norm.cdf(-abs(z))
    return 1 - 2 * nd / len(pairs), p


def kendall_in_parts(xs, ys):
    pairs_num = 0
    N = 0
    tmp = len(xs[0]) * (len(xs[0]) - 1) / 2
    # print(kendalltau(xs[0], ys[0]),
    #       1 - 2 * (kendall_mergesort(0, len(xs[0]), xs[0], ys[0]) / tmp))
    exchanges = []
    for sub_x, sub_y in zip(xs, ys):
        assert len(sub_x) == len(sub_y)
        n = len(sub_x)
        N += n
        pairs_num += n * (n - 1) / 2
        temp = list(range(n))
        # perm = list(range(len(sub_x)))
        perm = np.lexsort((sub_y, sub_x))
        # perm.sort(key=lambda a, b: cmp(
        #     sub_x[a], sub_x[b]) or cmp(sub_y[a], sub_y[b]))
        exchanges.append(kendall_mergesort(
            0, len(sub_x), sub_x, sub_y, perm, temp))
        # assert (abs(kendalltau(sub_x, sub_y)[0] -(1 -
        #       2 * exchanges[-1] / (n * (n - 1) / 2))) < 0.001)
    exchanges = sum(exchanges)
    tau = 1 - 2 * exchanges / pairs_num
    print("p-val is not accurate, N is too big")
    z = 3 * (pairs_num - 2 * exchanges) / math.sqrt((2 * N + 5) * pairs_num)
    p = 2 * norm.cdf(z)
    return tau, p

##########################################################################
# Project DB specific
##########################################################################


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
