from io import TextIOWrapper
import os
import itertools
import re
from datetime import datetime

from nltk.corpus import stopwords

RESOURCE_PATH = "../resources"
RESULT_PATH = "../results"

def get_sentences(source: TextIOWrapper) -> list[list[str]]:
    """
    Parameters
    ----------
    source: TextIOWrapper
        A file pointer pointing to a text file containing the sampled sentences.
    
    Returns
    -------
    list[list[str]]
        A list of lists, the elements of which are the lemma_pos tokens of each sentence.
    """

    sentences = []

    for line in source:
        sentences.append(line.strip().split(" "))

    return sentences

def reduce_corpus(source):
    """
    Transforms ITWAC from its original XML format to a more suitable format for this use case.
    1. Stopwords are removed.
    2. Only lemma and part of speech are considered.
    3. Only sentences with more than 10 tokens are considered (after removing the stopwords).
    4. The sentences are in horizontal format, as opposed to the original vertical format. The lemma and part of speech are separated by an underscore; the lemma_pos that compose a sentence are separated by a space. Each line contains a sentence.
    """
    doc = []
    inside = False

    # The patterns are compiled for performance reasons
    pattern_split = re.compile("\t")
    pattern_sub = re.compile("[\W\d_]")
    pattern_pos = re.compile(":") # Discards additional information about part of speech tags

    pos_tag = ("ADJ", "ADV", "NOUN", "VER") # The part of speech tags of interest
    stop_words = set(stopwords.words("italian"))
    n_docs = 0

    print("Reducing corpus...")

    # The sentences are contained between <s> tags and are in vertical format
    with open(f"{RESOURCE_PATH}/ITWAC/ITWAC_redux.txt", "w", encoding="utf-8") as out:
        for line in source:
            if line.startswith("</s"):
                if len(doc) > 10: # Discards documents shorter than 10 tokens
                    out.write(f'{" ".join(doc)}\n') # Makes the sentence horizontal and writes it on a single line

                    if n_docs % 100000 == 0: # Logs the number of sentences done
                        print(f"{datetime.now().strftime('%H:%M')}: {n_docs} done.")

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