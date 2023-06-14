from collections import defaultdict
from datetime import datetime
import re
import json
import os

from gensim.corpora import Dictionary
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon as j_distance
from gensim.models import HdpModel, LdaModel, LdaMulticore

from utils import RESOURCE_PATH, RESULT_PATH, get_sentences

def _build_base_topic_distribution(model_type: str, dct: Dictionary, save: bool = True) -> dict[str, dict[int, int]]:
    """
    Builds the base topic distribution from which the others will be derived.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP)
        - "lda": Latent Dirichlet Allocation (LDA) 
        - "ldamulti": LDA with multicore implementation
    
    dct: Dictionary
        The Gensim Dictionary to use to get the words.

    save: bool, default True
        Whether to save the results to file.

    Returns
    -------
    dict[str, dict[int, int]]
        A dictionary where the key is the lemma_pos and the value is another dictionary representing its topic distribution.
        The topic distribution is a dictionary where the key is the topic number and the value is its frequency.
    """
    lemma_td = {}
    mode = "base"

    for i, lemma_pos in enumerate(dct.values()):
        match model_type:
            case "hdp":
                model = HdpModel.load(f"{RESOURCE_PATH}/models/hdp/{lemma_pos}.dat")
            case "lda":
                model = LdaModel.load(f"{RESOURCE_PATH}/models/lda/{lemma_pos}.dat")
            case "ldamulti":
                model = LdaMulticore.load(f"{RESOURCE_PATH}/models/ldamulti/{lemma_pos}.dat")

        lemma, pos = lemma_pos.split("_")
        td = defaultdict(int)

        filename = f"{RESOURCE_PATH}/sentences/{pos}/{lemma}.txt"
        with open(filename, "r", encoding="latin-1") as infile:
            sentences = get_sentences(infile)

            for sentence in sentences:
                topic_prob_dist = model[dct.doc2bow(sentence)]

                if topic_prob_dist != []:
                    best_topic, _ = max(topic_prob_dist, key=lambda t: t[1])
                    td[best_topic] += 1

        print(f"{datetime.now().strftime('%H:%M')} {i+1}/{len(dct)}\t{lemma_pos}")
        lemma_td[lemma_pos] = len(td), td

    print(f"Built topic distribution for {model_type} ({mode}).")

    if save == True:
        with open(f"{RESULT_PATH}/topic_distribution_{model_type}_base.txt", "w", encoding="utf-8") as outfile:
            for lemma_pos, (n_senses, td) in lemma_td.items():
                outfile.write(f"{lemma_pos}\t{n_senses}\t{dict(td)}\n")

    return lemma_td

def get_topic_distribution(model_type: str, dct: Dictionary, mode: str = "base", save: bool = True) -> dict[str, dict[int, int]]:
    """
    Builds the topic distribution for the specified model type and mode.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP)
        - "lda": Latent Dirichlet Allocation (LDA) 
        - "ldamulti": LDA with multicore implementation

    dct: Dictionary
        The Gensim Dictionary to use to get the words.
    
    mode: str, default "base"
        The mode of the topic distribution. Possible values are the following:
        - "base": builds the topic distribution as is.
        - "jdist": builds the topic distribution and adjusts the values using the Jensen-Shannon distance.
        - "prune": builds the topic distribution and prunes the least frequent topics.

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
        return _build_base_topic_distribution(model_type, dct, save=save)

    if not os.path.exists(f"{RESULT_PATH}/topic_distribution_{model_type}_base.txt"):
        print("Error: build base distribution first.")
        return

    with open(f"{RESULT_PATH}/topic_distribution_{model_type}_base.txt", "r", encoding="utf-8") as file:
        for line in file:
            lemma_pos, n_senses, td = line.strip().split("\t")
            td = re.sub('(\d+):', r'"\1":', td)
            td = json.loads(td)
            base_td[lemma_pos] = int(n_senses), td

    for lemma_pos, (n_senses, td) in base_td.items():
        if mode == "prune": # Filter topics with < 0.15 prob
            td = dict(filter(lambda x: (x[1]/sum(td.values())) > 0.15, td.items())) 
            n_senses = len(td)
        elif mode == "jdist": # Adjusts the number of senses with Jensen-Shannon distance
            p = list(td.values())
            q = [1] * n_senses
            n_senses *= 1 - j_distance(p, q)

        lemma_td[lemma_pos] = n_senses, td

    print(f"Built topic distribution for {model_type} ({mode}).")

    if save == True:
        with open(f"{RESULT_PATH}/topic_distribution_{model_type}_{mode}.txt", "w", encoding="utf-8") as outfile:
            for lemma_pos, (n_senses, td) in lemma_td.items():
                outfile.write(f"{lemma_pos}\t{n_senses}\t{dict(td)}\n")

    return lemma_td

def get_correlation(model_type: str, mode: str, save: bool = True) -> dict[str, tuple[float, float]]:
    """
    Computes the correlation between the number of annotated senses and the number of induced senses for the specified model type and mode.

    Parameters
    ----------
    model_type: str
        The model of the topic model. Possible values are the following:
        - "hdp": Hierarchical Dirichlet Process (HDP)
        - "lda": Latent Dirichlet Allocation (LDA) 
        - "ldamulti": LDA with multicore implementation
    
    mode: str
        The mode of the topic distribution. Possible values are the following:
        - "base": builds the topic distribution as is.
        - "jdist": builds the topic distribution and adjusts the values using the Jensen-Shannon distance.
        - "prune": builds the topic distribution and prunes the least frequent topics.

    save: bool, default True
        Whether to save the results to file.

    Returns
    -------
    dict[str, tuple[float, float]]
        A dictionary whose keys are the parts of speech tags (plus "TOT" for the overall correlation) and the values are tuples where the first value is the Spearman coefficient for WordNet and the second one is the Spearman coefficient for Wiktionary.
    """
    with open(f"{RESOURCE_PATH}/stats/adj.txt", "r", encoding="utf-8") as adj, open(f"{RESOURCE_PATH}/stats/adv.txt", "r", encoding="utf-8") as adv, open(f"{RESOURCE_PATH}/stats/noun.txt", "r", encoding="utf-8") as noun,\
        open(f"{RESOURCE_PATH}/stats/verb.txt", "r", encoding="utf-8") as ver, open(f"{RESULT_PATH}/topic_distribution_{model_type}_{mode}.txt", "r", encoding="utf-8") as lemma_topic:

        print(f"Building correlation for {model_type} ({mode})...")

        pos_file = {
            "ADJ": adj,
            "ADV": adv,
            "NOUN": noun,
            "VER": ver,
        }

        lemma_pos_topic = {} # "lemma_pos": n_topic

        for line in lemma_topic:
            line = line.strip()
            lemma_pos, n_topic, _ = line.split("\t")
            lemma_pos_topic[lemma_pos] = float(n_topic)

        n_topics_wordnet_tot = []
        n_topics_wiktionary_tot = []
        n_senses_wordnet_tot = []
        n_senses_wiktionary_tot = []

        result = {}

        for pos, file in pos_file.items():
            lemma_senses = {} # "lemma_pos": (n_senses_wordnet, n_senses_wiktionary)

            for line in file:
                lemma, wordnet_senses, wiktionary_senses, _, _ = line.split("\t")
                lemma = f"{lemma}_{pos}"
                wordnet_senses = int(wordnet_senses)
                wiktionary_senses = int(wiktionary_senses)

                lemma_senses[lemma] = (wordnet_senses, wiktionary_senses)
            
            n_topics_wordnet = []
            n_topics_wiktionary = []
            n_senses_wordnet = []
            n_senses_wiktionary = []
            
            for lemma_pos, (n_wordnet, n_wiktionary) in lemma_senses.items():
                if lemma_pos in lemma_pos_topic:
                    n_topic = lemma_pos_topic[lemma_pos]

                    if n_wordnet != -1:
                        n_topics_wordnet.append(n_topic)
                        n_senses_wordnet.append(n_wordnet)

                    if n_wiktionary != -1:
                        n_topics_wiktionary.append(n_topic)
                        n_senses_wiktionary.append(n_wiktionary)

            coeff_wordnet = spearmanr(n_topics_wordnet, n_senses_wordnet)
            coeff_wiktionary = spearmanr(n_topics_wiktionary, n_senses_wiktionary)

            n_topics_wordnet_tot.extend(n_topics_wordnet)
            n_topics_wiktionary_tot.extend(n_topics_wiktionary)
            n_senses_wordnet_tot.extend(n_senses_wordnet)
            n_senses_wiktionary_tot.extend(n_senses_wiktionary)

            result[pos] = (coeff_wordnet, coeff_wiktionary)

        coeff_wordnet_tot = spearmanr(n_topics_wordnet_tot, n_senses_wordnet_tot)
        coeff_wiktionary_tot = spearmanr(n_topics_wiktionary_tot, n_senses_wiktionary_tot)

        result["TOT"] = (coeff_wordnet_tot, coeff_wiktionary_tot)

        print(f"Built correlation for {model_type} ({mode})")

        if save == True:
            with open(f"{RESULT_PATH}/corr_{model_type}_{mode}.txt", "w", encoding="utf-8") as corr:
                corr.write("POS\tWordnet-Correlation\tWordnet-pvalue"
                    "\tWiktionary-Correlation\tWiktionary-pvalue\n")
                for pos, (coeff_wordnet, coeff_wiktionary) in result.items():
                    corr.write(f"{pos}\t{coeff_wordnet.correlation}\t{coeff_wordnet.pvalue}"
                    f"\t{coeff_wiktionary.correlation}\t{coeff_wiktionary.pvalue}\n")

        return result