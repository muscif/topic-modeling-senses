import itertools
import os

from data_analysis import build_topic_distribution, build_correlation
from gensim.corpora import Dictionary
from utils import RESOURCE_PATH, LANGUAGE, reduce_corpus, get_language
from sample import build_dict, sample_dict, sample_sentences_naive
from topic_modeling import build_model
from words_search import build_senses

def do_all():
    match LANGUAGE:
        case "it":
            corpus = "ITWAC"
        case "en":
            corpus = "UKWAC"

    # The corpus is partitioned in multiple files, this makes it readable as a single file
    filenames = [filename for filename in os.listdir(f"{RESOURCE_PATH}/{LANGUAGE}/corpus") if filename.startswith(f"{corpus}-")]
    source = itertools.chain(*[open(f"{RESOURCE_PATH}/{LANGUAGE}/corpus/{filename}", "r", encoding="latin-1") for filename in filenames])
    reduce_corpus(source)

    build_senses()

    with open(f"{RESOURCE_PATH}/{LANGUAGE}/corpus/corpus_redux.txt", "r", encoding="utf-8") as infile:
        dct = build_dict(infile)
        dct = sample_dict(1000, dct)

        sample_sentences_naive(5000, infile, dct)

    for model in ["hdp", "lda"]:
        build_model(model, dct)

        for mode in ["base", "jdist", "prune", "pareto", "lrsum", "gradient", "kneedle"]:
            build_topic_distribution(model, dct, mode)
            build_correlation(model, mode)

if __name__=="__main__":
    dct = Dictionary.load(f"{RESOURCE_PATH}/{LANGUAGE}/dct.dat")
    dct = sample_dict(5000, dct)

    with open(f"{RESOURCE_PATH}/{LANGUAGE}/corpus/corpus_redux.txt", "r", encoding="utf-8") as infile:
        sample_sentences_naive(5000, infile, dct)

    for model in ["hdp", "lda"]:
        build_model(model, dct)

        for mode in ["base", "jdist", "prune", "pareto", "lrsum", "gradient", "kneedle"]:
            build_topic_distribution(model, dct, mode, multicore=False)
            build_correlation(model, mode)