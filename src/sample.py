from io import TextIOWrapper
import random
import re
from typing import Iterator

from gensim.corpora import Dictionary

from utils import RESOURCE_PATH, LANGUAGE, get_time

# Build dictionary
def build_dict(source: TextIOWrapper, save: bool = True) -> Dictionary:
    """
    Builds the Gensim dictionary by reading the source.

    Parameters
    ----------
    source: TextIOWrapper
        Corpus over which to build the dictionary.

    save: bool, default True
        Whether to save the results to file.

    Returns
    -------
    Dictionary
        A Gensim dictionary built on the provided corpus.
    """
    print("Building dictionary...")
    dct = Dictionary()
    tmp = Dictionary() # tmp dictionary is needed because only one variable uses too much memory
    pattern = re.compile(" ")

    for i, line in enumerate(source):
        doc = pattern.split(line.strip())

        tmp.add_documents([doc])

        if i % 100000 == 0:
            dct.merge_with(tmp)
            tmp = Dictionary()
            print(f"{get_time()} {i} merged.")

    dct.merge_with(tmp)
    dct.compactify()

    print("Built dictionary.")

    if save == True:
        dct.save(f"{RESOURCE_PATH}/{LANGUAGE}/dictionaries/dct.dat")

    return dct

def sample_dict(top_n: int, dct: Dictionary, save: bool = True) -> Dictionary:
    """
    Gets the top_n words in the dictionary that are in the sense dictionary.

    Parameters
    ----------
    top_n: int
        The number of top words to consider.

    dct: Dictionary
        Gensim dictionary to sample from.

    save: bool, default True
        Whether to save the results to file.

    Returns
    -------
    Dictionary
        The dictionary containing only the sampled words.
    """
    good_ids = set()
    lemmas = set()

    path_onto = f"{RESOURCE_PATH}/{LANGUAGE}/dictionaries/dict_onto.tsv"
    with open(path_onto, "r", encoding="utf-8") as infile:
        for line in infile:
            lemma_pos = line.split("\t")[0]
            lemmas.add(lemma_pos)

    for id, lemma_pos in dct.items():
        if lemma_pos in lemmas:
            good_ids.add(id)

    dct.filter_tokens(good_ids=good_ids)
    dct.filter_extremes(no_below=0, no_above=1, keep_n=top_n)

    if save == True:
        dct.save(f"{RESOURCE_PATH}/{LANGUAGE}/dictionaries/dct_top{top_n}.dat")

    return dct
    
# Sample k sentences containing words in the dictionary
# Reservoir sampling, algorithm R
def sample_sentences_reservoir(k: int, source: Iterator[list[str]], dct: Dictionary, save: bool = True) -> dict[str, list[list[str]]]:
    """
    DEPRECATED
    ----------
    Samples k sentences from the corpus for each lemma_pos in the dictionary using the technique of reservoir sampling.

    Parameters
    ----------
    k: int
        The number of sentences to sample.

    source: Iterator[list[str]]
        Iterator over lists of lemma_pos in the corpus, obtained by using extract_sentences.

    dct: Dictionary
        Gensim dictionary containing the lemma_pos.

    save: bool, default True
        Whether to save the results to file.

    Returns
    -------
    dict[str, list[list[str]]]
        A dictionary where the key is the lemma_pos and the value is the list of sentences containing it.
        
        The elements of the list are sentences, which are lists containing strings representing the lemma_pos.
    """
    lemmas = {lemma_pos: [] for lemma_pos in dct.values()}

    print("Building reservoir...")
    for sentence in source:
        for lemma_pos in sentence:
            if lemma_pos in lemmas and len(r := lemmas[lemma_pos]) < k:
                r.append(sentence.copy())

        if all(len(v) == k for v in lemmas.values()):
            break

    print("Built reservoir.")
    print("Sampling...")

    for i, sentence in enumerate(source, k):
        j = random.randrange(1, i)

        for lemma_pos in sentence:
            if lemma_pos in lemmas and j < k:
                lemmas[lemma_pos][j] = sentence.copy()

    print("Built sentences.")

    if save == True:
        for lemma_pos, sentences in lemmas.items():
            filename = f"{RESOURCE_PATH}/{LANGUAGE}/sentences/{lemma_pos}.txt"
            with open(filename, "w", encoding="utf-8") as outfile:
                for sentence in sentences:
                    outfile.write(f'{" ".join(sentence)}\n')
                    
            sentences = None
    
    return lemmas

def sample_sentences_naive(n: int, source: TextIOWrapper, dct: Dictionary, save: bool = True) -> dict[str, list[list[str]]]:
    """
    Samples n sentences from the corpus for each lemma_pos in the dictionary using a naive technique.

    Parameters
    ----------
    n: int
        The number of sentences to sample.

    source: TextIOWrapper
        The corpus from which to sample sentences.

    dct: Dictionary
        Gensim dictionary containing the lemma_pos.

    save: bool, default True
        Whether to save the results to file.

    Returns
    -------
    dict[str, list[list[str]]]
        A dictionary where the key is the lemma_pos and the value is the list of sentences containing it. The elements of the list are sentences, which are lists containing strings representing the lemma_pos.
    """
    lemmas = dict((lemma_pos, set()) for lemma_pos in dct.values())

    print("Building sentences...")
    source = list(source)
    pattern_split = re.compile(" ")

    # Get indexes where each sentence is present
    for i, sentence in enumerate(source):
        sentence = pattern_split.split(sentence)

        for lemma_pos in sentence:
            if lemma_pos in lemmas:
                lemmas[lemma_pos].add(i)

    print("Built sentences.")

    if save == True:
        print("Saving sentences...")

        # Get the sentences based on the indices
        for lemma_pos in dct.values():
            with open(f"{RESOURCE_PATH}/{LANGUAGE}/sentences/{lemma_pos}.txt", "w", encoding="utf-8") as outfile:
                indexes = lemmas[lemma_pos]

                if len(indexes) > n:
                    indexes = random.sample(list(indexes), n)

                for index in indexes:
                    sentence = source[index]
                    outfile.write(sentence)

        print("Saved sentences.")

    return lemmas