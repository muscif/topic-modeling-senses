import random
import re
from io import TextIOWrapper
from typing import Iterator

from gensim.corpora import Dictionary

from config import PATH_DICTIONARIES, PATH_SENTENCES, WINDOW_SIZE
from utils import get_time


def _get_window(l: list, element) -> list:
    """
    Gets a window centered around the provided element.
    
    Parameters
    ----------
    l: list
        The list to get the window from.

    element
        The element to center the window on.

    Returns
    -------
    list
        A list centered around the provided element.
    """
    i = l.index(element)

    left = i - WINDOW_SIZE // 2
    right = i + 1 + WINDOW_SIZE // 2

    if left < 0:
        right += abs(0 - left)
        left = 0
    elif right > len(l):
        left -= right - len(l)
        right = len(l)

    return l[left:right]

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
    tmp = Dictionary()
    pattern = re.compile(" ")

    for i, line in enumerate(source):
        doc = pattern.split(line.strip())
        tmp.add_documents([doc])

        if i % 1000000 == 0:
            dct.merge_with(tmp)
            tmp = Dictionary()
            print(f"{get_time()} {i} merged.")

    dct.merge_with(tmp)
    dct.compactify()

    print("Built dictionary.")

    if save is True:
        dct.save(f"{PATH_DICTIONARIES}/dct.dat")

    return dct

def sample_dict(top_n: int, n_sentences: int, dct: Dictionary, save: bool = True) -> Dictionary:
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
    onto_lemmas = set()

    path_onto = f"{PATH_DICTIONARIES}/dct_onto.tsv"
    with open(path_onto, "r", encoding="utf-8") as infile:
        # 
        for line in infile:
            lemma_pos = line.split("\t")[0]
            onto_lemmas.add(lemma_pos)

    good_ids = set()

    for id, lemma_pos in dct.items():
        if lemma_pos in onto_lemmas:
            good_ids.add(id)

    dct.filter_tokens(good_ids=good_ids)
    dct.filter_extremes(no_below=n_sentences, no_above=1, keep_n=top_n)

    if save is True:
        dct.save(f"{PATH_DICTIONARIES}/dct_top{top_n}_sample{n_sentences}.dat")

    return dct

def sample_dict_foreach(top_n: int, n_sentences: int, dct: Dictionary, save: bool = True) -> Dictionary:
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
    onto_lemmas = set()

    path_onto = f"{PATH_DICTIONARIES}/dct_onto.tsv"
    with open(path_onto, "r", encoding="utf-8") as infile:
        for line in infile:
            lemma_pos = line.split("\t")[0]
            onto_lemmas.add(lemma_pos)

    good_ids = set()

    for id, lemma_pos in dct.items():
        if lemma_pos in onto_lemmas:
            good_ids.add(id)

    dct.filter_tokens(good_ids=good_ids)

    pos_ids = dict((pos, set()) for pos in ["ADJ", "ADV", "NOUN", "VER"])

    for pos, ids in pos_ids.items():
        id = set(k for k in dct if dct[k].endswith(pos))
        id = list(sorted(id, key=lambda x: dct.dfs[x], reverse=True))[:top_n]
        ids.update(id)

    good_ids = set().union(*pos_ids.values())

    dct.filter_tokens(good_ids=good_ids)

    if save is True:
        dct.save(f"{PATH_DICTIONARIES}/dct_top{top_n}foreach_sample{n_sentences}.dat")

    return dct

def sample_dict_onto(dct: Dictionary, save: bool = True):
    """
    Gets from the dictionary only the words present in the ontologies.
    
    Parameters
    ----------
    dct: Dictionary
        Gensim dictionary to sample from.

    save: bool, default True
        Whether to save the results to file.

    Returns
    -------
    Dictionary
        The dictionary containing only the sampled words.
    """
    onto_lemmas = set()

    path_onto = f"{PATH_DICTIONARIES}/dct_onto.tsv"
    with open(path_onto, "r", encoding="utf-8") as infile:
        for line in infile:
            lemma_pos = line.split("\t")[0]
            onto_lemmas.add(lemma_pos)

    good_ids = set()

    for id, lemma_pos in dct.items():
        if lemma_pos in onto_lemmas:
            good_ids.add(id)

    dct.filter_tokens(good_ids=good_ids)

    if save is True:
        dct.save(f"{PATH_DICTIONARIES}/dct_onto.dat")

    return dct

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

    if save is True:
        for lemma_pos, sentences in lemmas.items():
            filename = f"{PATH_SENTENCES}/{lemma_pos}.txt"
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

    if save is True:
        print("Saving sentences...")

        # Get the sentences based on the indices
        for lemma_pos in dct.values():
            with open(f"{PATH_SENTENCES}/{lemma_pos}.txt", "w", encoding="utf-8") as outfile:
                indexes = lemmas[lemma_pos]

                if len(indexes) > n:
                    indexes = random.sample(list(indexes), n)

                for index in indexes:
                    sentence = source[index]

                    doc = sentence.strip().split(" ")
                    doc = _get_window(doc, lemma_pos)

                    sentence = " ".join(doc)
                    outfile.write(sentence + "\n")

        print("Saved sentences.")

    return lemmas
