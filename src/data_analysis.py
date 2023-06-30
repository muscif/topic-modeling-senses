from collections import Counter
from pickle import UnpicklingError
import re
import json
import os
from multiprocessing import cpu_count, Pool

from gensim.corpora import Dictionary
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon as j_distance
from gensim.models import HdpModel, LdaModel
import numpy as np
from kneed import KneeLocator

from utils import RESOURCE_PATH, RESULT_PATH, LANGUAGE, get_sentences, split_list, get_process_number, get_time

def _build_base_topic_distribution_worker(model_type: str, dct: Dictionary, todo: list):
    """
    Worker for building the base topic distribution.

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

    lemma_td: dict[str, dict[int, float]]
        The dictionary containing the topic distribution for each lemma.
    """
    t_number = get_process_number()
    print(f"T{t_number} started.")

    match model_type:
        case "hdp":
            load_model = HdpModel.load
        case "lda":
            load_model = LdaModel.load
    
    lemma_td = {}

    for i, lemma_pos in enumerate(todo):
        try:
            model = load_model(f"{RESOURCE_PATH}/{LANGUAGE}/models/{model_type}/{lemma_pos}.dat")
        except UnpicklingError:
            print(f"{lemma_pos} not found.")
            continue
        
        acc = Counter()
        sentences = get_sentences(lemma_pos)

        for sentence in sentences:
            topic_prob_dist = model[dct.doc2bow(sentence)]

            if topic_prob_dist != []:
                acc.update(dict(topic_prob_dist))

        acc = dict(acc)
        lemma_td[lemma_pos] = acc

        print(f"{get_time()} T{t_number} {i+1}/{len(todo)} {lemma_pos}")

    print(f"T{t_number} ended.")

    return lemma_td

def _build_base_topic_distribution(model_type: str, dct: Dictionary, save: bool = True, multicore: bool = False) -> dict[str, dict[int, float]]:
    """
    Builds the base topic distribution from which the others will be derived.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP)
        - "lda": Latent Dirichlet Allocation (LDA)
    
    dct: Dictionary
        The Gensim Dictionary to use to get the words.

    save: bool, default True
        Whether to save the results to file.

    multicore: bool, default True
        Whether to compute on multiple cores or a single one. If True, the number of processes equals n_cores - 1. It's currently broken and causes a BSOD for unknown reasons.

    Returns
    -------
    dict[str, dict[int, int]]
        A dictionary where the key is the lemma_pos and the value is another dictionary representing its topic distribution.
        The topic distribution is a dictionary where the key is the topic number and the value is its frequency.
    """
    mode = "base"
    
    if multicore == True:
        n_process = cpu_count() - 1
        todos = split_list(list(dct.values()), n_process)
        
        with Pool(n_process) as pool:
            res = pool.starmap(_build_base_topic_distribution_worker, [(model_type, dct, todo) for todo in todos])

        dict_total = {}
        for d in res:
            dict_total.update(d)

        res = dict_total
    else:
        todo = list(dct.values())
        res = _build_base_topic_distribution_worker(model_type, dct, todo)

    if save == True:
        filename = f"{RESULT_PATH}/{LANGUAGE}/{model_type}/td_base.txt"
        with open(filename, "w", encoding="utf-8") as outfile:
            for lemma_pos, td in res.items():
                n_senses = len(td)
                td = dict((k, v/sum(td.values())) for k, v in td.items())
                td = dict(sorted(td.items(), key=lambda x: x[1], reverse=True))
                outfile.write(f"{lemma_pos}\t{n_senses}\t{td}\n")

    print(f"Built topic distribution for {model_type} ({mode}).")

def build_topic_distribution(model_type: str, dct: Dictionary, mode: str, multicore: bool = False, save: bool = True) -> dict[str, dict[int, int]]:
    """
    Builds the topic distribution for the specified model type and mode.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP)
        - "lda": Latent Dirichlet Allocation (LDA)

    dct: Dictionary
        The Gensim Dictionary to use to get the words.
    
    mode: str, default "base"
        The mode of the topic distribution. Possible values are the following:
        - "base": builds the base topic distribution as is.
        - "prune": adjusts the base topic distribution using the Jensen-Shannon distance.
        - "jdist": prunes the least frequent (<0.15) topics.
        - "pareto": cuts senses based on the Pareto (20/80) principle
        - "lrsum": equilibrium index which minimizes the difference
        - "gradient": finds the elbow by finding the maximum point of the second derivative
        - "kneedle": finds the elbow by using the Kneedle algorithm

    multicore: bool, default False
        Whether to compute on multiple cores or a single one. If True, the number of processes equals n_cores - 1. It's currently broken and causes a BSOD for unknown reasons.

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
        return _build_base_topic_distribution(model_type, dct, multicore=multicore)

    filename = f"{RESULT_PATH}/{LANGUAGE}/{model_type}/td_base.txt"
    if not os.path.exists(filename):
        print("Error: build base distribution first.")
        return

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            lemma_pos, n_senses, td = line.strip().split("\t")
            td = re.sub('(\d+):', r'"\1":', td)
            td = json.loads(td)
            base_td[lemma_pos] = int(n_senses), td

    # These assume that base_td is sorted by value (decreasing)
    for lemma_pos, (n_senses, td) in base_td.items():
        match mode:
            case "prune": # Filter topics with < 0.15 prob
                #td = {(k, v) for (k, v) in td.items() if td[k]/sum(td.values()) > 0.15}
                td = dict(filter(lambda x: (x[1]/sum(td.values())) > 0.15, td.items())) 
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
                    n_senses = max_n if max_n > 2 else 2
                    td = dict(list(td.items())[:n_senses])
            case "kneedle": # Finds the elbow by using the Kneedle algorithm
                if n_senses > 4:
                    x = np.arange(len(td))
                    y = np.asarray(list(td.values()))
                    kneedle = KneeLocator(x, y, S=1.0, curve="convex", direction="decreasing")
                    n_senses = kneedle.elbow
                    td = dict(list(td.items())[:n_senses])

        lemma_td[lemma_pos] = n_senses, td

    print(f"Built topic distribution for {model_type} ({mode}).")

    if save == True:
        filename = f"{RESULT_PATH}/{LANGUAGE}/{model_type}/td_{mode}.txt"
        with open(filename, "w", encoding="utf-8") as outfile:
            for lemma_pos, (n_senses, td) in lemma_td.items():
                outfile.write(f"{lemma_pos}\t{n_senses}\t{dict(td)}\n")

    return lemma_td

def build_correlation(model_type: str, mode: str, save: bool = True) -> dict[str, tuple[float, float]]:
    """
    Computes the correlation between the number of annotated senses and the number of induced senses for the specified model type and mode.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP)
        - "lda": Latent Dirichlet Allocation (LDA)
    
    mode: str, default "base"
        The mode of the topic distribution. Possible values are the following:
        - "base": builds the base topic distribution as is.
        - "prune": adjusts the base topic distribution using the Jensen-Shannon distance.
        - "jdist": prunes the least frequent (<0.15) topics.
        - "pareto": cuts senses based on the Pareto (20/80) principle
        - "lrsum": equilibrium index which minimizes the difference
        - "gradient": finds the elbow by finding the maximum point of the second derivative
        - "kneedle": finds the elbow by using the Kneedle algorithm

    save: bool, default True
        Whether to save the results to file.

    Returns
    -------
    dict[str, tuple[float, float]]
        A dictionary whose keys are the parts of speech tags (plus "TOT" for the overall correlation) and the values are tuples where the first value is the Spearman coefficient for WordNet and the second one is the Spearman coefficient for Wiktionary.
    """
    print(f"Building correlation for {model_type} ({mode})...")

    path_dict = f"{RESOURCE_PATH}/{LANGUAGE}/dictionaries/dict_onto.tsv"
    path_topic_distribution = f"{RESULT_PATH}/{LANGUAGE}/{model_type}/td_{mode}.txt"

    # Get number of topics for each lemma
    lemma_topic = {}
    with open(path_topic_distribution, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            lemma_pos, n_topic, _ = line.split("\t")
            lemma_topic[lemma_pos] = float(n_topic)

    # Get number of senses for each lemma
    lemma_senses = {}
    with open(path_dict, "r", encoding="utf-8") as infile:
        for line in infile:
            lemma_pos, senses_wnet, senses_wikt = line.strip().split("\t")
            senses_wnet = int(senses_wnet)
            senses_wikt = int(senses_wikt)
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

    if save == True:
        filename = f"{RESULT_PATH}/{LANGUAGE}/{model_type}/corr_{mode}.tsv"

        with open(filename, "w", encoding="utf-8") as corr:
            corr.write("POS\tWordnet-Correlation\tWordnet-pvalue\tWiktionary-Correlation\tWiktionary-pvalue\n")

            for pos, (coeff_wnet, coeff_wikt) in result.items():
                corr.write(f"{pos}\t{coeff_wnet.correlation}\t{coeff_wnet.pvalue}\t{coeff_wikt.correlation}\t{coeff_wikt.pvalue}\n")

    return result