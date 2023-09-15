import os
import re
from datetime import datetime
from io import TextIOWrapper
from multiprocessing import cpu_count, current_process

from nltk.corpus import stopwords
from numpy import array_split

from config import LANGUAGE, PATH_CORPUS, PATH_DICTIONARIES, PATH_RESULT, PATH_SENTENCES, WINDOW_SIZE


def _get_stopwords() -> set[str]:
    """
    Returns
    -------
    set[str]
        The set of stopword for current language.
    """
    match LANGUAGE:
        case "it":
            lang = "italian"
        case "en":
            lang = "english"

    a = set(stopwords.words(lang))

    with open(f"{PATH_DICTIONARIES}/stopwords.txt", "r", encoding="utf-8") as infile:
        b = set(line.strip() for line in infile)

    return a | b

def get_time():
    """
    Returns
    -------
    str
        The current time formatted as hh:mm.
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

    filename = f"{PATH_SENTENCES}/{lemma_pos}.txt"
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
        The list to be splitted.

    n: int
        The number of sublists to get.

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

def get_total_processes(workers: int) -> int:
    """
    Determines the number of processes to use based on the provided workers.

    Parameters
    ----------
    workers: int
        The number of desired workers.

    Returns
    -------
    The number of processes to spawn.
    """
    if workers == -1:
        n_process = cpu_count() - 1
    elif workers > 1:
        if workers > cpu_count() - 1:
            print("Error: workers can't be greater than the number of cores")
        else:
            n_process = workers
    else:
        n_process = 1

    return n_process

def reduce_corpus(source: TextIOWrapper):
    """
    Transforms the corpus from its original XML format to a more suitable format for this use case.
    1. Stopwords are removed.
    2. Only the lemma and part of speech are considered.
    3. Only sentences with more than 10 tokens are considered (after removing the stopwords).
    4. The sentences are in horizontal format, as opposed to the original vertical format. The lemma and part of speech are separated by an underscore; the lemma_pos that compose a sentence are separated by a space. Each line contains a sentence.
    """

    # The patterns are compiled for performance reasons
    pattern_split = re.compile("\t")
    pattern_sub = re.compile(r"[\W\d_]")
    # Discards additional information about part of speech tags
    pattern_pos = re.compile(":")

    # The tags are in tuple form so that str.startswith can be used with multiple tags
    match LANGUAGE:
        case "en":
            pos_tag = ("J", "R", "N", "V")
        case "it":
            pos_tag = ("ADJ", "ADV", "NOUN", "VER")

    # POS translation from English to Italian
    pos_translation = {
        "J": "ADJ",
        "R": "ADV",
        "N": "NOUN",
        "V": "VER"
    }

    stop_words = _get_stopwords()
    n_docs = 0

    doc = []
    inside = False

    print("Reducing corpus...")

    hashes = set()

    # The sentences are contained between <s> tags and are in vertical format
    filename = f"{PATH_CORPUS}/corpus_redux.txt"
    with open(filename, "w", encoding="utf-8") as out:
        for line in source:
            # Executes when the sentence ends
            if line.startswith("</s"):
                # Discards documents shorter than 10 tokens
                if len(doc) > WINDOW_SIZE:
                    # Makes the sentence horizontal and writes it on a single line
                    sentence = " ".join(doc)
                    sentence_hash = hash(sentence)

                    # Checks whether the sentence has been written before
                    if sentence_hash not in hashes:
                        hashes.add(sentence_hash)
                        out.write(f"{sentence}\n")

                        # Logs the number of sentences done
                        if n_docs % 100000 == 0:
                            print(f"{get_time()}: {n_docs} done.")

                        n_docs += 1

                doc.clear()
                inside = False

            if inside:
                tokens = pattern_split.split(line.strip().lower())

                if len(tokens) != 3:
                    continue

                word, pos, lemma = tokens
                pos = pos.upper()

                if pos.startswith(pos_tag) and word not in stop_words and lemma not in stop_words:
                    lemma = pattern_sub.sub("", lemma)

                    if len(lemma) > 1:
                        pos = pattern_pos.split(pos)[0]
                        pos = pattern_sub.sub("", pos)

                        if LANGUAGE != "it":
                            pos = pos_translation[pos[0]] # First letter

                        doc.append(f"{lemma}_{pos}")

            if line.startswith("<s"):
                inside = True

    print("Reduced corpus.")

def print_good(model_type: str, save: bool = True):
    """
    Prints significant results for the specified model_type.
    A result is considered significant if the absolute value of the correlation is > 0.2 and the p-value is < 0.05

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP).
        - "lda": Latent Dirichlet Allocation (LDA).

    save: bool, default True
        Whether to save the results.
    """
    files = [filename for filename in os.listdir(f"{PATH_RESULT}/{model_type}") if filename.endswith(".tsv")]
    good = []
    
    # Thresholds
    corr_th = 0.2
    pv_th = 0.05

    for file in files:
        with open(f"{PATH_RESULT}/{model_type}/{file}", "r", encoding="utf-8") as infile:
            infile.readline() # Skip header

            for line in infile:
                pos, wnet_c, wnet_p, wikt_c, wikt_p = line.strip().split("\t")

                wnet_c = float(wnet_c)
                wnet_p = float(wnet_p)
                wikt_c = float(wikt_c)
                wikt_p = float(wikt_p)

                mode = file.split(".")[0]
                mode = mode.split("_")[1]

                if abs(wnet_c) > corr_th and wnet_p < pv_th:
                    good.append((mode, "wnet", pos, str(wnet_c), str(wnet_p)))
                if abs(wikt_c) > corr_th and wikt_p < pv_th:
                    good.append((mode, "wikt", pos, str(wikt_c), str(wikt_p)))

    print(f"Good results for {LANGUAGE}/{model_type}:")
    print("mode\tonto\tpos\tcorrelation\t\tp-value")

    for e in good:
        print("\t".join(e))

    if save is True:
        with open(f"{PATH_RESULT}/{model_type}/good_results.txt", "w", encoding="utf-8") as outfile:
            outfile.write("mode\tonto\tpos\tcorrelation\t\tp-value\n")

            for e in good:
                outfile.write("\t".join(e) + "\n")

def create_folders():
    """
    Creates the folders used by the project.
    """
    if "resources" not in os.listdir("../"):
        os.mkdir("../resources")

    if LANGUAGE not in os.listdir("../resources"):
        os.mkdir(f"../resources/{LANGUAGE}")

    for fold in ["corpus", "dictionaries", "models", "sentences"]:
        if fold not in os.listdir(f"../resources/{LANGUAGE}"):
            os.mkdir(f"../resources/{LANGUAGE}/{fold}")

    for model in ["hdp", "lda"]:
        if model not in os.listdir(f"../resources/{LANGUAGE}/models/"):
            os.mkdir(f"../resources/{LANGUAGE}/models/{model}")
