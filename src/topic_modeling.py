import os
from datetime import datetime

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import HdpModel, LdaModel, LdaMulticore
import numpy as np
from multiprocessing import cpu_count, Pool, current_process

from utils import RESOURCE_PATH, get_sentences

def build_model(model_type: str, dct: Dictionary, save: bool = True, resume: bool = True) -> None:
    """
    Builds the topic model for each lemma_pos in the dictionary, where the corpus is the sampled sentences for that word.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP)
        - "lda": Latent Dirichlet Allocation (LDA) 
        - "ldamulti": LDA with multicore implementation
    
    dct: Dictionary
        The Gensim Dictionary to use to build the topic model.
    
    save: bool, default True
        Whether to save the results to file.
    
    resume: bool, default True
        Whether to build the models just for the remaining words or to start over.
    """
    print(f"Building {model_type} model...")

    words = dct.values()

    if resume:
        done = os.listdir(f"{RESOURCE_PATH}/models/{model_type}")
        done = list(filter(lambda x: x.endswith(".dat"), done))
        to_build = [lp for lp in words if lp not in map(lambda x: x.split(".")[0], done)]
    else:
        done = []
        to_build = words

    for i, lemma_pos in enumerate(to_build, len(done)):
        lemma, pos = lemma_pos.split("_")
        filename = f"{RESOURCE_PATH}/sentences/{pos}/{lemma}.txt"

        with open(filename, "r", encoding="latin-1") as infile:
            sentences = get_sentences(infile)
            corpus = [dct.doc2bow(doc) for doc in sentences]
            best_umass = -100
            best_model = 0

            for n in range(2, 20):
                match model_type:
                    case "hdp":
                        model = HdpModel(corpus=corpus, id2word=dct, T=n, random_state=101)
                    case "lda":
                        model = LdaModel(corpus=corpus, id2word=dct, num_topics=n, random_state=101)
                    case "ldamulti":
                        model = LdaMulticore(corpus=corpus, id2word=dct, num_topics=n, random_state=101)

                cm = CoherenceModel(model=model, corpus=corpus, dictionary=dct, coherence="u_mass")
                cs = cm.get_coherence()

                if cs > best_umass:
                    best_umass = cs
                    best_model = model

            if save == True:
                best_model.save(f"{RESOURCE_PATH}/models/{model_type}/{lemma_pos}.dat")

            print(f"{datetime.now().strftime('%H:%M')} {i+1}/{len(words)}")

def _core_computation(model_type: str, todo: list, dct: Dictionary, save: bool = True) -> None:
    t_name = current_process().name
    t_name = t_name.replace("SpawnPoolWorker-", "")
    print(f"T{t_name} started.")

    for i, lemma_pos in enumerate(todo):
        lemma, pos = lemma_pos.split("_")
        filename = f"{RESOURCE_PATH}/sentences/{pos}/{lemma}.txt"

        with open(filename, "r", encoding="latin-1") as infile:
            sentences = get_sentences(infile)
            corpus = [dct.doc2bow(doc) for doc in sentences]
            best_umass = -100
            best_model = 0

            for n in range(2, 20):
                match model_type:
                    case "hdp":
                        model = HdpModel(corpus=corpus, id2word=dct, T=n, random_state=101)
                    case "lda":
                        model = LdaModel(corpus=corpus, id2word=dct, num_topics=n, random_state=101)

                cm = CoherenceModel(model=model, corpus=corpus, dictionary=dct, coherence="u_mass")
                cs = cm.get_coherence()

                if cs > best_umass:
                    best_umass = cs
                    best_model = model

            if save == True:
                best_model.save(f"{RESOURCE_PATH}/models/{model_type}/{lemma_pos}.dat")

            print(f"{datetime.now().strftime('%H:%M')} {t_name} - {i+1}/{len(todo)}")

def _split_list(l: list, n: int) -> list[list]:
    splits = np.array_split(l, n)
    return [list(a) for a in splits]

def build_model_multicore(model_type: str, dct: Dictionary, save: bool = True, resume: bool = True) -> None:
    """
    Builds the topic model for each lemma_pos in the dictionary, where the corpus is the sampled sentences for that word.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP)
        - "lda": Latent Dirichlet Allocation (LDA) 
    
    dct: Dictionary
        The Gensim Dictionary to use to build the topic model.
    
    save: bool, default True
        Whether to save the results to file.
    """
    print(f"Building {model_type} model...")
    n_thread = cpu_count() - 1

    if resume == True:
        l = os.listdir(f"{RESOURCE_PATH}/models/{model_type}")
        l = set(dct.values()) - set(map(lambda x: x.strip(".dat"), l))
        words = _split_list(list(l), n_thread)
    else:
        words = _split_list(list(dct.values()), n_thread)
    
    #l = lambda word: _core_computation(model_type, word, dct, save=save)
    model_types = [model_type] * n_thread
    dcts = [dct] * n_thread
    saves = [save] * n_thread

    pool = Pool(n_thread)
    pool.starmap(_core_computation, zip(model_types, words, dcts, saves))