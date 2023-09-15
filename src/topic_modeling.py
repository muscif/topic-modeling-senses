import os
from multiprocessing import Pool
from typing import Iterable

from gensim.corpora import Dictionary
from gensim.models import HdpModel, LdaModel
from gensim.models.coherencemodel import CoherenceModel

from config import PATH_MODEL
from utils import get_process_number, get_sentences, get_time, get_total_processes, split_list

def _build_model_worker(model_type: str, todo: Iterable[str], dct: Dictionary, save: bool):
    """
    Worker for building the topic models.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP).
        - "lda": Latent Dirichlet Allocation (LDA).

    todo: Iterable[str]
        The list of lemmas to process.

    dct: Dictionary
        The Gensim Dictionary to use to get the words.

    save: bool, default True
        Whether to save the results to file.
    """
    t_number = get_process_number()
    print(f"P{t_number} started.")

    for i, lemma_pos in enumerate(todo):
        sentences = get_sentences(lemma_pos)
        corpus = [dct.doc2bow(doc) for doc in sentences]

        match model_type:
            case "hdp":
                best_model = HdpModel(corpus=corpus, id2word=dct, T=30, alpha=0.001, eta=0.001, random_state=101)
                print(f"{get_time()} P{t_number} - {i+1}/{len(todo)} {lemma_pos}")
            case "lda":
                best_umass = -100

                for i in range(1, 20):
                    model = LdaModel(corpus=corpus, id2word=dct, num_topics=i, alpha=0.001, eta=0.001, random_state=101)
                    umass = CoherenceModel(model=model, corpus=corpus, coherence="u_mass").get_coherence()

                    if umass > best_umass:
                        best_umass = umass
                        best_model = model

                print(f"{get_time()} P{t_number} - {i+1}/{len(todo)} {lemma_pos} {best_umass} n={best_model.num_topics}")

        if save is True:
            best_model.save(f"{PATH_MODEL}/{model_type}/{lemma_pos}.dat")

def build_model(model_type: str, dct: Dictionary, todo: Iterable[str], workers: int = -1, resume: bool = True, save: bool = True) -> None:
    """
    Builds the topic model for each lemma_pos in the dictionary, where the corpus is the sampled sentences for that word.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP).
        - "lda": Latent Dirichlet Allocation (LDA).

    dct: Dictionary
        The Gensim Dictionary to use to build the topic model.

    todo: list
        The list of lemmas to process.

    workers: int, default -1
        The number of processes to spawn.

    resume: bool, default True
        Whether to resume the computation or to start again.

    save: bool, default True
        Whether to save the results to file.
    """
    print(f"Building {model_type} model...")

    if resume is True:
        l = os.listdir(f"{PATH_MODEL}/{model_type}")
        todo = list(x for x in todo if f"{x}.dat" not in l)
    else:
        todo = list(todo)

    n_process = get_total_processes(workers)

    if n_process > 1:
        todos = split_list(todo, n_process)

        with Pool(n_process) as pool:
            pool.starmap(_build_model_worker, [(model_type, todo, dct, save) for todo in todos])
    else:
        _build_model_worker(model_type, todo, dct, save)
