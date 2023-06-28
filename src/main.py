import itertools
import os

from data_analysis import get_topic_distribution, get_correlation
from gensim.corpora import Dictionary
from utils import RESOURCE_PATH, reduce_corpus
from sample import extract_sentences, build_dict, sample_dict, sample_sentences_naive, sample_sentences_reservoir, sample_dict_stats
from topic_modeling import build_model

def do_all(sentences_sample_mode, dict_sample_mode):
    # The corpus is partitioned in multiple files, this makes it readable as a single file
    filenames = [filename for filename in os.listdir(f"{RESOURCE_PATH}/ITWAC") if filename.startswith("ITWAC-")]
    source = itertools.chain(*[open(f"{RESOURCE_PATH}/ITWAC/{filename}", "r", encoding="latin-1") for filename in filenames])
    reduce_corpus(source)

    with open(f"{RESOURCE_PATH}/corpus/ITWAC_redux.txt", "r", encoding="utf-8") as infile:
        source = extract_sentences(infile)
        dct = build_dict(source)
        
        match dict_sample_mode:
            case "base":
                dct = sample_dict(5000, 1000, dct)
            case "stats":
                dct = sample_dict_stats(dct)

        match sentences_sample_mode:
            case "reservoir":
                sample_sentences_reservoir(5000, source, dct)
            case "naive":
                sample_sentences_naive(5000, source, dct)

    for model in ["hdp", "lda"]:
        build_model(model, dct)

        for mode in ["base", "jdist", "prune", "pareto", "lrsum", "gradient", "kneedle"]:
            get_topic_distribution(model, dct, mode)
            get_correlation(model, mode)

if __name__=="__main__":
    dct = Dictionary.load(f"{RESOURCE_PATH}/dictionaries/dct_stats.dat")

    for model in ["hdp", "lda"]:
        for mode in ["base", "jdist", "prune", "pareto", "lrsum", "gradient", "kneedle"]:
            get_topic_distribution(model, dct, mode, multicore=True)
            get_correlation(model, mode)