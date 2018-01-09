import numpy as np
import os
import sys
import re
import itertools

PROJECT = os.path.realpath(os.path.dirname(__file__)) + os.sep
ANNOTATION_FILE = os.path.join(
    PROJECT, "conll14st-test-data", "alt", "official-2014.combined-withalt.m2")
NOOP = "noop"

print("check apply_changes functions")


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


def create_ranks(file, max_permutations=100, filter_annot_changes=lambda x: True, ignore_noop=True, max_changes=None):
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
    for source, all_changes in db:
        sentence_variations = []
        if ignore_noop:

            if find_in_iter(all_changes, NOOP):
                continue
        for annot_changes in all_changes:
            annot_changes = list(filter(filter_annot_changes, annot_changes))
            total_annotations += 1
            for permutation_id, changes in zip(range(max_permutations), itertools.permutations(annot_changes, max_changes)):
                rank = []
                for i in range(len(annot_changes) + 1):
                    rank.append(
                        (apply_changes(source, changes[:i]), changes[:i]))
                    total_sentences += 1
                    # if total_sentences % 1000 == 0:
                    #     print ("created a total of", total_sentences, "sentences")
                sentence_variations.append(rank)
        if len(sentence_variations) > 1:
            ranks.append(sentence_variations)
            if len(ranks) % 10 == 0:
                print("calculated for", len(ranks), "source sentences")
    print("Created", total_sentences, "sentences based on", len(ranks),
          "eligible sentences and a total of", total_annotations, "annotations.")
    return ranks


def create_levelled_files(ranks, file_num):
    """ creates parallel files by the order of the NUCLE sentences, choosing annotators randomly"""
    print("function was not checked")
    files = []
    for i in range(file_num):
        file = []
        for sentence_variations in ranks:
            sentences = sentence_variations[
                np.random.randint(len(sentence_variations))]
            corrections_num = min(i, len(sentences) - 1)
            line = sentences[corrections_num][0]
            file.append(line)
        files.append(file)
    return files


def main():
    max_permutations = 100
    ranks = create_ranks(ANNOTATION_FILE)
    # print(ranks[0][:2])
    print([x[:2] for x in create_levelled_files(ranks, 5)]


if __name__ == '__main__':
    main()
