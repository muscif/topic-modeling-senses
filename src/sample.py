from dataclasses import dataclass
from io import TextIOWrapper
import random
import re
from datetime import datetime
from typing import Generator, Iterator

from gensim.corpora import Dictionary

from utils import RESOURCE_PATH

# Extract sentences from corpus
def extract_sentences(source: TextIOWrapper) -> list[str]:
    """
    Iterates over the sentences of the corpus.

    Parameters
    ----------
    source: TextIOWrapper
        The source file for the corpus in the reduced format described in utils.reduce_corpus.

    Yields
    ------
    list[str]
        List of lemma_pos for each sentence.
    """
    pattern = re.compile(" ")

    for line in source:
        yield pattern.split(line.strip())

# Build dictionary
def build_dict(source: Iterator[list[str]], save: bool = True) -> Dictionary:
    """
    Builds the Gensim dictionary by reading the source.

    Parameters
    ----------
    source: Iterator[list[str]]
        Iterator over lists of lemma_pos in the corpus, obtained by using extract_sentences.

    save: bool, default True
        Whether to save the results to file.

    Returns
    -------
    Dictionary
        A Gensim dictionary built on the provided corpus.
    """
    print("Building dictionary...")
    dictionary = Dictionary()
    tmp = Dictionary() # tmp dictionary is needed because only one variable uses too much memory

    for i, doc in enumerate(source):
        tmp.add_documents([doc])

        if i % 100000 == 0:
            dictionary.merge_with(tmp)
            tmp = Dictionary()
            print(f"{datetime.now().strftime('%H:%M')}: {i} merged.")

    dictionary.merge_with(tmp)
    dictionary.compactify()

    print("Built dictionary.")

    if save == True:
        dictionary.save(f"{RESOURCE_PATH}/dct.dat")

    return dictionary

# Sample sample_n items from the top top_n items in dictionary
def sample_dict(top_n: int, sample_n: int, dictionary: Dictionary, save: bool = True) -> Dictionary:
    """
    Samples sample_n words from the top_n words in the dictionary.

    Parameters
    ----------
    top_n: int
        The number of top words to consider.
    
    sample_n: int
        The number of words to sample from the top_n.

    dictionary: Dictionary
        Gensim dictionary containing the words to be sampled.

    save: bool, default True
        Whether to save the results to file.

    Returns
    -------
    Dictionary
        The dictionary containing only the sampled words.
    """
    dictionary.filter_extremes(no_below=0, no_above=1, keep_n=top_n)
    good_ids = random.sample(range(top_n), sample_n)
    dictionary.filter_tokens(good_ids=good_ids)

    if save == True:
        dictionary.save(f"{RESOURCE_PATH}/dct_sample{sample_n}_top{top_n}.dat")

    return dictionary

# Sample k sentences containing words in the dictionary
# Reservoir sampling, algorithm R
def sample_sentences_reservoir(k: int, source: Iterator[list[str]], dct: Dictionary, save: bool = True) -> dict[str, list[list[str]]]:
    """
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
        A dictionary where the key is the lemma_pos and the value is the list of sentences containing it. The elements of the list are sentences, which are lists containing strings representing the lemma_pos.
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
            lemma, pos = lemma_pos.split("_")

            with open(f"{RESOURCE_PATH}/sentences/{pos}/{lemma}.txt", "w", encoding="latin-1") as outfile:
                for sentence in sentences:
                    outfile.write(f'{" ".join(sentence)}\n')
    
    return lemmas

def sample_sentences_naive(n: int, source: Iterator[list[str]], dct: Dictionary, save: bool = True) -> dict[str, list[list[str]]]:
    """
    Samples n sentences from the corpus for each lemma_pos in the dictionary using a naive technique.

    WARNING: uses up to 10GB of memory.

    Parameters
    ----------
    n: int
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
        A dictionary where the key is the lemma_pos and the value is the list of sentences containing it. The elements of the list are sentences, which are lists containing strings representing the lemma_pos.
    """
    lemmas = {}

    @dataclass
    class Lemma:
        count: int
        indexes: set
        sentences: set

    print("Initializing...")

    for k, v in dct.items():
        indexes = set(random.sample(range(0, dct.dfs[k]), n))
        lemmas[v] = Lemma(0, indexes, set())

    print("Building sentences...")
    for sentence in source:
        for lemma_pos in sentence:
            if lemma_pos in lemmas:
                l = lemmas[lemma_pos]

                if l.count in l.indexes:
                    l.sentences.add(tuple(sentence))

                l.count += 1
                lemmas[lemma_pos] = l

    print("Built sentences.")

    if save == True:
        for lemma_pos, l in lemmas.items():
            lemma, pos = lemma_pos.split("_")

            with open(f"{RESOURCE_PATH}/sentences/{pos}/{lemma}.txt", "w", encoding="latin-1") as outfile:
                for sentence in l.sentences:
                    outfile.write(f'{" ".join(sentence)}\n')

    return lemmas