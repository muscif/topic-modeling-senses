from io import TextIOWrapper
import re
from datetime import datetime
from multiprocessing import current_process
import os

from nltk.corpus import stopwords
from numpy import array_split

RESOURCE_PATH = "../resources"
RESULT_PATH = "../results"
LANGUAGE = "it"

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

    filename = f"{RESOURCE_PATH}/{LANGUAGE}/sentences/{lemma_pos}.txt"
    with open(filename, "r", encoding="utf-8") as infile:
        for line in infile:
            sentences.append(line.strip().split(" "))

    return sentences

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
    
    match LANGUAGE:
        case "en":
            lang = "english"
            pos_tag = ("J", "R", "N", "V")
        case "it":
            lang = "italian"
            pos_tag = ("ADJ", "ADV", "NOUN", "VER")
    
    stop_words = set(stopwords.words(lang))
    n_docs = 0

    pos_translation = {
        "J": "ADJ",
        "R": "ADV",
        "N": "NOUN",
        "V": "VER"
    }

    doc = []
    inside = False

    print("Reducing corpus...")

    # The sentences are contained between <s> tags and are in vertical format
    filename = f"{RESOURCE_PATH}/{LANGUAGE}/corpus/corpus_redux.txt"
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

                    if pos.startswith(pos_tag) and not (word in stop_words or lemma in stop_words):
                        lemma = pattern_sub.sub("", lemma)
                        pos = pattern_sub.sub("", pos)

                        if LANGUAGE != "it":
                            pos = pos_translation[pos[0]]

                        if len(lemma) > 1:
                            doc.append(f"{lemma}_{pos}")

            if line.startswith("<s"):
                inside = True

    print("Reduced corpus.")

def get_language() -> str:
    """
    Returns
    -------
    The set language.
    """
    return LANGUAGE

def print_good(model_type: str):
    """
    Prints significant results for the specified model_type.
    A result is considered significant if the absolute value of the correlation is > 0.2 and the p-value is < 0.05

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP)
        - "lda": Latent Dirichlet Allocation (LDA)
    """
    files = [filename for filename in os.listdir(f"{RESULT_PATH}/{LANGUAGE}/{model_type}") if filename.endswith(".tsv")]
    good = []

    for file in files:
        with open(f"{RESULT_PATH}/{LANGUAGE}/{model_type}/{file}", "r", encoding="utf-8") as infile:
            infile.readline() # Skip header
            
            for line in infile:
                pos, wnet_c, wnet_p, wikt_c, wikt_p = line.strip().split("\t")
                
                wnet_c = float(wnet_c)
                wnet_p = float(wnet_p)
                wikt_c = float(wikt_c)
                wikt_p = float(wikt_p)
                
                if abs(wnet_c) > 0.2 and wnet_p < 0.05:
                    good.append((file.split(".")[0], "wnet", pos, str(wnet_c), str(wnet_p)))
                if abs(wikt_c) > 0.2 and wikt_p < 0.05:
                    good.append((file.split(".")[0], "wikt", pos, str(wikt_c), str(wikt_p)))

    print("Good results:")
    print("mode\t\tonto\tpos\tcorrelation\t\tp-value")

    for e in good:
        print("\t".join(e))