from io import TextIOWrapper
import re
from datetime import datetime
from multiprocessing import current_process

from nltk.corpus import stopwords
from numpy import array_split

RESOURCE_PATH = "../resources"
RESULT_PATH = "../results"

def get_time():
    """
    Returns
    -------
    str
        The current time formatted as hh:mm
    """
    return datetime.now().strftime('%H:%M')

def get_sentences(lemma_pos: str) -> list[list[str]]:
    """
    Parameters
    ----------
    lemma_pos: str
        The lemma_pos of which to get the sampled sentences.
    
    Returns
    -------
    list[list[str]]
        A list of lists, the elements of which are the lemma_pos tokens of each sentence.
    """

    sentences = []
    lemma, pos = lemma_pos.split("_")

    filename = f"{RESOURCE_PATH}/sentences/{pos}/{lemma}.txt"
    with open(filename, "r", encoding="latin-1") as infile:
        for line in infile:
            sentences.append(line.strip().split(" "))

    return sentences

def reduce_corpus(source: TextIOWrapper):
    """
    Transforms ITWAC from its original XML format to a more suitable format for this use case.
    1. Stopwords are removed.
    2. Only lemma and part of speech are considered.
    3. Only sentences with more than 10 tokens are considered (after removing the stopwords).
    4. The sentences are in horizontal format, as opposed to the original vertical format. The lemma and part of speech are separated by an underscore; the lemma_pos that compose a sentence are separated by a space. Each line contains a sentence.
    """
    # The patterns are compiled for performance reasons
    pattern_split = re.compile("\t")
    pattern_sub = re.compile("[\W\d_]")
    # Discards additional information about part of speech tags
    pattern_pos = re.compile(":")

    # The part of speech tags of interest
    pos_tag = ("ADJ", "ADV", "NOUN", "VER")
    stop_words = set(stopwords.words("italian"))
    n_docs = 0

    doc = []
    inside = False

    print("Reducing corpus...")

    # The sentences are contained between <s> tags and are in vertical format
    filename = f"{RESOURCE_PATH}/corpus/ITWAC_redux.txt"
    with open(filename, "w", encoding="utf-8") as out:
        for line in source:
            if line.startswith("</s"):
                # Discards documents shorter than 10 tokens
                if len(doc) > 10:
                    # Makes the sentence horizontal and writes it on a single line
                    out.write(f'{" ".join(doc)}\n')

                    # Logs the number of sentences done
                    if n_docs % 100000 == 0:
                        print(f"{get_time()}: {n_docs} done.")

                    n_docs += 1
                
                doc.clear()
                inside = False

            if inside:
                tokens = pattern_split.split(line.strip())

                if len(tokens) == 3:
                    word, pos, lemma = tokens
                    word = word.lower()
                    pos = pattern_pos.split(pos)[0]
                    lemma = lemma.lower()

                    if pos in pos_tag and not (word in stop_words or lemma in stop_words):
                        lemma = pattern_sub.sub("", lemma)
                        pos = pattern_sub.sub("", pos)

                        if len(lemma) > 1:
                            doc.append(f"{lemma}_{pos}")

            if line.startswith("<s"):
                inside = True

    print("Reduced corpus.")

def split_list(l: list, n: int) -> list[list]:
    """
    Splits l into n sub-lists of (approximately) equal length.

    Parameters
    ----------
    l: list
        The list to be splitted

    n: int
        The number of sublists to get

    Returns
    -------
    list[list]
        A list of n lists.
    """
    splits = array_split(l, n)

    # This step is necessary to transform an ndarray in a regular list
    splits = [list(a) for a in splits]
    return splits

def get_process_number() -> int:
    """
    Returns
    -------
    int
        The number of the calling process.
    """
    t_name = current_process().name

    if t_name == "MainProcess":
        return 0

    t_name = t_name.replace("SpawnPoolWorker-", "")
    return int(t_name)