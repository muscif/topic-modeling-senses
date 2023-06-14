import itertools
import os

from gensim.corpora import Dictionary

from data_analysis import get_topic_distribution, get_correlation
from utils import RESOURCE_PATH, reduce_corpus
from sample import extract_sentences, build_dict, sample_dict, sample_sentences_naive, sample_sentences_reservoir
from topic_modeling import build_model, build_model_multicore

def do_all(sample_mode):
    # The corpus is partitioned in multiple files, this makes it readable as a single file
    filenames = [filename for filename in os.listdir(f"{RESOURCE_PATH}/ITWAC") if filename.startswith("ITWAC-")]
    source = itertools.chain(*[open(f"{RESOURCE_PATH}/ITWAC/{filename}", "r", encoding="latin-1") for filename in filenames])
    reduce_corpus(source)

    with open(f"{RESOURCE_PATH}/ITWAC/ITWAC_redux.txt", "r", encoding="utf-8") as infile:
        source = extract_sentences(infile)
        dct = build_dict(source)
        dct = sample_dict(5000, 1000, dct)

        match sample_mode:
            case "reservoir":
                sample_sentences_reservoir(5000, source, dct)
            case "naive":
                sample_sentences_naive(5000, source, dct)

    for model in ["hdp", "lda", "ldamulti"]:
        build_model(model, dct)

        for mode in ["base", "jdist", "prune"]:
            get_topic_distribution(model, dct, mode)
            get_correlation(model, mode)

if __name__=="__main__":
    dct = Dictionary.load(f"{RESOURCE_PATH}/dct_sample1000_top5000.dat")
    
    for model in ["lda", "ldamulti"]:
        build_model(model, dct)
            
        for mode in ["base", "jdist", "prune"]:
            get_topic_distribution(model, dct, mode)
            get_correlation(model, mode)