import itertools
import os

from config import LANGUAGE, PATH_CORPUS, PATH_DICTIONARIES
from data_analysis import build_correlation, build_topic_distribution
from sample import build_dict, sample_sentences_naive, sample_dict
from topic_modeling import build_model
from utils import create_folders, print_good, reduce_corpus
from words_search import build_senses


def do_all():
    """
    Runs the whole project from start to finish.
    """
    create_folders()

    match LANGUAGE:
        case "it":
            corpus = "ITWAC"
        case "en":
            corpus = "UKWAC"

    # The corpus is partitioned in multiple files, this line makes it readable as a single file
    filenames = [filename for filename in os.listdir(PATH_CORPUS) if filename.startswith(f"{corpus}-") and filename.endswith(".xml")]
    files = [open(f"{PATH_CORPUS}/{filename}", "r", encoding="latin-1") for filename in filenames]
    source = itertools.chain(*files)

    reduce_corpus(source)

    for file in files:
        file.close()

    build_senses()

    with open(f"{PATH_CORPUS}/corpus_redux.txt", "r", encoding="utf-8") as infile:
        dct = build_dict(infile)
        dct = sample_dict(1000, 5000, dct)

        sample_sentences_naive(5000, infile, dct)

    todo = dct.values()

    for model in ["hdp", "lda"]:
        build_model(model, dct, todo)

        for mode in ["base", "jdist", "prune", "pareto", "lrsum", "gradient", "kneedle"]:
            build_topic_distribution(model, mode, todo)
            build_correlation(model, mode, todo)

        print_good(model)


if __name__ == "__main__":
    from gensim.corpora import Dictionary

    dct = Dictionary.load(f"{PATH_DICTIONARIES}/dct_top5000_sample5000.dat")

    todo = dct.values()

    for model in ["hdp", "lda"]:
        build_model(model, dct, todo)

        for mode in ["base", "jdist", "prune", "pareto", "lrsum", "gradient", "kneedle"]:
            build_topic_distribution(model, mode, todo)
            build_correlation(model, mode, todo)

        print_good(model)
