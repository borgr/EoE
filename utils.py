import os
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
