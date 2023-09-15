import ast
import json
import os
import re
from collections import Counter
from multiprocessing import Pool
from pickle import UnpicklingError
from typing import Iterable

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import HdpModel, LdaModel
from kneed import KneeLocator
from scipy.spatial.distance import jensenshannon as j_distance
from scipy.stats import spearmanr

from config import PATH_DICTIONARIES, PATH_MODEL, PATH_RESULT
from utils import get_process_number, get_sentences, get_time, get_total_processes, split_list


def _build_base_topic_distribution_worker(model_type: str, todo: list):
    """
    Worker for building the base topic distribution.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP).
        - "lda": Latent Dirichlet Allocation (LDA).

    todo: list
        The list of lemmas to process.
    """
    t_number = get_process_number()
    print(f"P{t_number} started.")
    
    lemma_td = {}

    match model_type:
        case "hdp":
            load_model = HdpModel.load
        case "lda":
            load_model = LdaModel.load

    with open(f"{PATH_RESULT}/{model_type}/td_base_P{t_number}.tmp", "a", encoding="utf-8") as outfile:
        for i, lemma_pos in enumerate(todo):
            try:
                model = load_model(f"{PATH_MODEL}/{model_type}/{lemma_pos}.dat")
            except UnpicklingError as e:
                print(e)
                continue

            match model_type:
                case "hdp":
                    corpus = model.corpus
                    model = model.suggested_lda_model()
                case "lda":
                    dct = Dictionary.load(f"{PATH_MODEL}/{model_type}/{lemma_pos}.id2word")
                    corpus = [dct.doc2bow(s) for s in get_sentences(lemma_pos)]

            acc = Counter()

            for doc in corpus:
                topic_prob_dist = model.get_document_topics(doc, minimum_probability=0.0)
                acc.update(dict(topic_prob_dist))

            acc = dict(acc)
            lemma_td[lemma_pos] = acc

            outfile.write(f"{lemma_pos}\t{len(acc)}\t{acc}\n")

            print(f"{get_time()} P{t_number} {i+1}/{len(todo)} {lemma_pos}")

    print(f"P{t_number} ended.")

    return lemma_td

def _build_base_topic_distribution(model_type: str, todo: list, workers: int = -1, resume: bool = True, save: bool = True,) -> dict[str, dict[int, float]]:
    """
    Builds the base topic distribution from which the others will be derived.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP).
        - "lda": Latent Dirichlet Allocation (LDA).
    
    todo: list
        The list of lemmas to process.

    workers: int, default -1
        The number of processes to spawn.

    save: bool, default True
        Whether to save the results to file.
    
    Returns
    -------
    dict[str, dict[int, int]]
        A dictionary where the key is the lemma_pos and the value is another dictionary representing its topic distribution.
        The topic distribution is a dictionary where the key is the topic number and the value is its frequency.
    """
    mode = "base"
    tmp_files = [f for f in os.listdir(f"{PATH_RESULT}/{model_type}") if f.endswith(".tmp")]

    if resume is True:
        done = set()

        for file in tmp_files:
            with open(f"{PATH_RESULT}/{model_type}/{file}", "r", encoding="utf-8") as infile:
                for line in infile:
                    lemma_pos = line.split("\t")[0]
                    done.add(lemma_pos)

        todo = [e for e in todo if e not in done]

    n_process = get_total_processes(workers)

    if n_process > 1:
        todo = split_list(todo, n_process)

        with Pool(n_process) as pool:
            res = pool.starmap(_build_base_topic_distribution_worker, [(model_type, todo) for todo in todo])

        dict_total = dict()
        for d in res:
            dict_total.update(d)

        res = dict_total
    else:
        res = _build_base_topic_distribution_worker(model_type, todo)

    if save is True:
        filename = f"{PATH_RESULT}/{model_type}/td_base.txt"
        with open(filename, "w", encoding="utf-8") as outfile:
            for file in tmp_files:
                tmp_filename = f"{PATH_RESULT}/{model_type}/{file}"
                with open(tmp_filename, "r", encoding="utf-8") as infile:
                    for line in infile:
                        lemma_pos, _, td = line.strip().split("\t")
                        td = ast.literal_eval(td)
                        res[lemma_pos] = td

                #os.remove(tmp_filename)

            for lemma_pos, td in res.items():
                n_senses = len(td)
                td = dict((k, v/sum(td.values())) for k, v in td.items())
                td = dict(sorted(td.items(), key=lambda x: x[1], reverse=True))
                outfile.write(f"{lemma_pos}\t{n_senses}\t{td}\n")

    print(f"Built topic distribution for {model_type} ({mode}).")

def build_topic_distribution(model_type: str, mode: str, todo: Iterable[str], workers: int = -1, save: bool = True) -> dict[str, dict[int, int]]:
    """
    Builds the topic distribution for the specified model type and mode.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP).
        - "lda": Latent Dirichlet Allocation (LDA).

    mode: str, default "base"
        The mode of the topic distribution. Possible values are the following:
        - "base": builds the base topic distribution as is.
        - "prune": adjusts the base topic distribution using the Jensen-Shannon distance.
        - "jdist": prunes the least frequent (<0.15) topics.
        - "pareto": cuts senses based on the Pareto (20/80) principle.
        - "lrsum": equilibrium index which minimizes the difference.
        - "gradient": finds the elbow by finding the maximum point of the second derivative.
        - "kneedle": finds the elbow by using the Kneedle algorithm.

    todo: Iterable[str]
        The list of lemmas to process.

    workers: int, default -1
        The number of processes to spawn.

    save: bool, default True
        Whether to save the results to file.

    Returns
    -------
    dict[str, dict[int, int]]
        A dictionary where the key is the lemma_pos and the value is another dictionary representing its topic distribution.
        The topic distribution is a dictionary where the key is the topic number and the value is its frequency.
    """
    print(f"Building topic distribution for {model_type} ({mode})...")

    lemma_td = {}
    base_td = {}

    if mode == "base":
        return _build_base_topic_distribution(model_type, todo, workers=workers)

    filename = f"{PATH_RESULT}/{model_type}/td_base.txt"
    if not os.path.exists(filename):
        print("Error: build base distribution first.")
        return

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            lemma_pos, n_senses, td = line.strip().split("\t")
            td = re.sub(r'(\d+):', r'"\1":', td)
            td = json.loads(td)
            base_td[lemma_pos] = int(n_senses), td

    # These assume that base_td is sorted by value (decreasing)
    for lemma_pos, (n_senses, td) in base_td.items():
        match mode:
            case "prune": # Filter topics with < 0.15 prob
                #td = dict((k, v) for (k, v) in td.items() if td[k]/sum(td.values()) > 0.15)
                td = dict(filter(lambda x: (x[1]/sum(td.values())) > 0.01, td.items()))
                n_senses = len(td)
            
            case "jdist": # Adjusts the number of senses with Jensen-Shannon distance
                p = list(td.values())
                q = [1] * n_senses
                n_senses *= 1 - j_distance(p, q)
            
            case "pareto": # Cuts senses based on the Pareto (20/80) principle
                if n_senses > 2:
                    l = list(td.values())
                    i = min(range(len(td)), key=lambda i: abs(sum(l[:i]) - 0.80))
                    td = dict(list(td.items())[:i])
                    n_senses = len(td)
            
            case "lrsum": # Equilibrium index which minimizes the difference
                if n_senses > 2:
                    l = list(td.values())
                    i = min(range(len(l)), key=lambda i: abs(sum(l[:i]) - sum(l[i:])))
                    n_senses = i
                    td = dict(list(td.items())[:n_senses])
            
            case "gradient": # Finds the elbow by finding the maximum point of the second derivative
                if n_senses > 1:
                    d = np.gradient(list(td.values())) # First derivative
                    dd = np.gradient(d) # Second derivative
                    max_n = np.argmax(dd)
                    n_senses = max_n
                    td = dict(list(td.items())[:n_senses])
            
            case "kneedle": # Finds the elbow by using the Kneedle algorithm
                x = np.arange(len(td))
                y = np.asarray(list(td.values()))
                kneedle = KneeLocator(x, y, S=1.0, curve="convex", direction="decreasing")
                n_senses = kneedle.elbow

                if n_senses is None:
                    n_senses = len(td)
                
                td = dict(list(td.items())[:n_senses])

        lemma_td[lemma_pos] = n_senses, td

    print(f"Built topic distribution for {model_type} ({mode}).")

    if save is True:
        filename = f"{PATH_RESULT}/{model_type}/td_{mode}.txt"
        with open(filename, "w", encoding="utf-8") as outfile:
            for lemma_pos, (n_senses, td) in lemma_td.items():
                outfile.write(f"{lemma_pos}\t{n_senses}\t{dict(td)}\n")

    return lemma_td

def build_correlation(model_type: str, mode: str, todo: Iterable[str], save: bool = True) -> dict[str, tuple[float, float]]:
    """
    Computes the correlation between the number of annotated senses and the number of induced senses for the specified model type and mode.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP).
        - "lda": Latent Dirichlet Allocation (LDA).
    
    mode: str, default "base"
        The mode of the topic distribution. Possible values are the following:
        - "base": builds the base topic distribution as is.
        - "prune": adjusts the base topic distribution using the Jensen-Shannon distance.
        - "jdist": prunes the least frequent (<0.15) topics.
        - "pareto": cuts senses based on the Pareto (20/80) principle.
        - "lrsum": equilibrium index which minimizes the difference.
        - "gradient": finds the elbow by finding the maximum point of the second derivative.
        - "kneedle": finds the elbow by using the Kneedle algorithm.

    save: bool, default True
        Whether to save the results to file.

    Returns
    -------
    dict[str, tuple[float, float]]
        A dictionary whose keys are the parts of speech tags (plus "TOT" for the overall correlation) and the values are tuples where the first value is the Spearman coefficient for WordNet and the second one is the Spearman coefficient for Wiktionary.
    """
    print(f"Building correlation for {model_type} ({mode})...")

    path_dict = f"{PATH_DICTIONARIES}/dct_onto.tsv"
    path_topic_distribution = f"{PATH_RESULT}/{model_type}/td_{mode}.txt"

    # Get number of topics for each lemma
    lemma_topic = {}
    with open(path_topic_distribution, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            lemma_pos, n_topic, _ = line.split("\t")

            if lemma_pos in todo:
                lemma_topic[lemma_pos] = float(n_topic)

    # Get number of senses for each lemma
    lemma_senses = {}
    with open(path_dict, "r", encoding="utf-8") as infile:
        for line in infile:
            lemma_pos, senses_wnet, senses_wikt = line.strip().split("\t")
            senses_wnet = int(senses_wnet)
            senses_wikt = int(senses_wikt)

            if lemma_pos in todo:
                lemma_senses[lemma_pos] = senses_wnet, senses_wikt

    result = {}

    # Construct the arrays for use in spearmanr
    tot_topics = []
    tot_senses_wnet = []
    tot_senses_wikt = []
    
    # Build correlation for each POS
    for pos in ["ADJ", "ADV", "NOUN", "VER"]:
        topics = []
        senses_wnet = []
        senses_wikt = []
        
        for lemma_pos, (s_wnet, s_wikt) in lemma_senses.items():
            if lemma_pos.endswith(pos) and lemma_pos in lemma_topic:
                n_topic = lemma_topic[lemma_pos]

                topics.append(n_topic)
                senses_wnet.append(s_wnet)
                senses_wikt.append(s_wikt)

        coeff_wnet = spearmanr(topics, senses_wnet)
        coeff_wikt = spearmanr(topics, senses_wikt)

        tot_topics.extend(topics)
        tot_senses_wnet.extend(senses_wnet)
        tot_senses_wikt.extend(senses_wikt)

        result[pos] = (coeff_wnet, coeff_wikt)

    # Build total correlation
    coeff_wnet_tot = spearmanr(tot_topics, tot_senses_wnet)
    coeff_wikt_tot = spearmanr(tot_topics, tot_senses_wikt)

    result["TOT"] = (coeff_wnet_tot, coeff_wikt_tot)

    print(f"Built correlation for {model_type} ({mode}).")

    if save is True:
        filename = f"{PATH_RESULT}/{model_type}/corr_{mode}.tsv"

        with open(filename, "w", encoding="utf-8") as corr:
            corr.write("POS\tWordnet-Correlation\tWordnet-pvalue\tWiktionary-Correlation\tWiktionary-pvalue\n")

            for pos, (coeff_wnet, coeff_wikt) in result.items():
                corr.write(f"{pos}\t{coeff_wnet.correlation}\t{coeff_wnet.pvalue}\t{coeff_wikt.correlation}\t{coeff_wikt.pvalue}\n")

    return result
