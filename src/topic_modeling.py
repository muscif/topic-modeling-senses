import os

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import HdpModel, LdaModel
from multiprocessing import cpu_count, Pool

from utils import RESOURCE_PATH, get_sentences, split_list, get_process_number, get_time

def _build_model_worker(model_type: str, todo: list, dct: Dictionary, save: bool = True):
    """
    Worker for building the topic models.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP)
        - "lda": Latent Dirichlet Allocation (LDA)

    todo: list
        The list of lemmas to process.

    dct: Dictionary
        The Gensim Dictionary to use to get the words.

    save: bool, default True
        Whether to save the results to file.
    """
    t_number = get_process_number()
    print(f"T{t_number} started.")

    for i, lemma_pos in enumerate(todo):
        sentences = get_sentences(lemma_pos)
        corpus = [dct.doc2bow(doc) for doc in sentences]

        match model_type:
            case "hdp":
                best_model = HdpModel(corpus=corpus, id2word=dct, T=30, random_state=101)
                cm = CoherenceModel(model=best_model, corpus=corpus, dictionary=dct, coherence="u_mass")
                best_umass = cm.get_coherence()
                print(f"{get_time()} T{t_number} - {i+1}/{len(todo)} {lemma_pos} {best_umass}")
            case "lda":
                best_umass = -100

                for n in range(1, 15):
                    if model_type == "lda":
                        model = LdaModel(corpus=corpus, id2word=dct, num_topics=n, eta="auto", chunksize=5000, random_state=101)

                    cm = CoherenceModel(model=model, corpus=corpus, dictionary=dct, coherence="u_mass")
                    cs = cm.get_coherence()

                    if cs > best_umass:
                        best_umass = cs
                        best_model = model
                        best_n = n

                    print(f"{get_time()} T{t_number} - {i+1}/{len(todo)} {lemma_pos} {best_umass} n={best_n}")

        if save == True:
            best_model.save(f"{RESOURCE_PATH}/models/{model_type}/{lemma_pos}.dat")

def build_model(model_type: str, dct: Dictionary, save: bool = True, resume: bool = True, multicore: bool = True) -> None:
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

    resume: bool, default True
        Whether to resume the computation or to start again.

    multicore: bool, default True
        Whether to compute on multiple cores or a single one. If True, the number of processes equals n_cores - 1.
    """
    print(f"Building {model_type} model...")

    if resume == True:
        l = os.listdir(f"{RESOURCE_PATH}/models/{model_type}")
        todo = list(x for x in dct.values() if f"{x}.dat" not in l)
    else:
        todo = list(dct.values())

    if multicore == True:
        n_process = cpu_count() - 1
        todos = split_list(todo, n_process)

        with Pool(n_process) as pool:
            pool.starmap(_build_model_worker, [(model_type, todo, dct, save) for todo in todos])
    else:
        _build_model_worker(model_type, todo, dct, save)