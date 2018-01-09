import numpy as np

from utils import *
from create_db import create_ranks, iterate_chains

def choose_source_per_chain(ranks, chooser):
    return [chooser(chain) for chain in iterate_chains(ranks)]

def choose_source_per_sentence(ranks, chooser):
    sources = []
    for sentence_chains in ranks:
        annot_chains = sentence_chains[np.random.randint(len(sentence_chains))]
        chain = annot_chains[np.random.randint(len(annot_chains))]
        sources.append(chooser(chain))
    return sources

def extract_references_per_sentence(ids, reference_files):


def extract_references_per_chain(ranks, ids, reference_files):

    ids = [tple[1] for tple in iterate_chains(ranks, ids)]

def rerank(sources, references, ranks, sentence_measure, corpus_measure=None):
    """reranks the corpus by the measure of choice. returns the rerank and the rankings
       measure - a function that gets (source, references_iterable, sentence) and a returns a number
       if a corpus measure is given, all sources\references_iterables\sentences would be passed to it as an iterable instead of being evaluated one by one by sentence_measure."""
    if corpus_measure is not None:
        scores_it = "not implemented"
        raise "unimplemented yet"
    scores = []
    for source, references, chain in zip(sources, references, iterate_chains(ranks)):
        chain_scores = []
        for sentence in chain:
            if corpus_measure is None:
                score = sentence_measure(source, references, sentence)
            else:
                score = next(scores_it)
            chain_scores.append(score)
        scores.append(chain_scores)
    return scores


def main():
    max_permutations = 1
    filename = str(max_permutations) + "_" + "rank" + ".json"
    ids_filename = os.path.join(CACHE_DIR,  "id" + filename)
    ranks_filename = os.path.join(CACHE_DIR,  "id" + filename)
    ranks = create_ranks(ANNOTATION_FILE, max_permutations, ranks_out_file=ranks_filename, ids_out_file=ids_filename)

    bleu_rank = rerank(sources, references, ranks, lambda so, r, sy: BLEU_score(
        so, r, sy, 4, nltk.translate.bleu_score.SmoothingFunction().method3, lambda x: x))
    # print(ranks[0][:2])
    # print([x[:2] for x in create_levelled_files(ranks, 5)])

if __name__ == '__main__':
    main()
