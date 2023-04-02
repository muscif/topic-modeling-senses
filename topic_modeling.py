import logging
import itertools
import subprocess
from time import time
import os
import re
import random
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from pprint import pp
from random import sample

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models import HdpModel, LdaModel, LdaMulticore
from nltk.corpus import stopwords
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon as j_distance

import json

pattern_split = re.compile("\t")
pattern_sub = re.compile("[\W\d_]")
pattern_pos = re.compile(":")
pos_tag = ("ADJ", "ADV", "NOUN", "VER")
stop_words = set(stopwords.words("italian"))
dct: Dictionary = Dictionary().load("top1000_dictionary.dat")
N_SENTENCES = 68147862

# Extract sentences from corpus
def extract_sentences(source):
    doc = []
    inside = False

    for line in source:
        if line.startswith("</s"):
            yield doc
            doc.clear()
            inside = False

        if inside:
            tokens = line.strip().split("\t")

            if len(tokens) == 3:
                word, pos, lemma = tokens
                word = word.lower()
                pos = pos.split(":")[0]
                lemma = lemma.lower()

                if pos in pos_tag and not (word in stop_words or lemma in stop_words):
                    lemma = pattern_sub.sub("", lemma)
                    pos = pattern_sub.sub("", pos)

                    if len(lemma) > 1:
                        doc.append(f"{lemma}_{pos}")

        if line.startswith("<s"):
            inside = True

# Get sampled sentences
def get_sentences(source):
    doc = []
    sentences = []

    for line in source:
        if line != "\n":
            lemma_pos = line.strip().replace("\t", "_")
            doc.append(lemma_pos)
        else:
            sentences.append(doc.copy())
            doc.clear()

    return sentences

# Sample k sentences containing words in the dictionary
# Reservoir sampling, algorithm R
def sample_sentences(k, source, save=True):
    lemmas = {lemma_pos: [] for lemma_pos in dct.values()}

    print("Building reservoir...")
    for sentence in source:
        for lemma_pos in sentence:
            if lemma_pos in lemmas and len(r := lemmas[lemma_pos]) < k and len(sentence) > 30:
                r.append(sentence.copy())

        if all(len(v) == k for v in lemmas.values()):
            break

    print("Sampling...")

    for i, sentence in enumerate(source, k):
        j = random.randrange(1, i)

        for lemma_pos in sentence:
            if lemma_pos in lemmas and j < k:
                lemmas[lemma_pos][j] = sentence.copy()

    if save == True:
        for lemma_pos, sentences in lemmas.items():
            lemma, pos = lemma_pos.split("_")

            with open(f"Sentences_vertical/{pos}/{lemma}.txt", "w", encoding="latin-1") as outfile:
                for sentence in sentences:
                    outfile.write(f'{" ".join(sentence)}\n')
    
    return lemmas

# Build dictionary
def build_dict(source):
    print("Building dictionary...")
    dictionary = Dictionary()
    tmp = Dictionary() # tmp dictionary is needed because only one variable uses too much memory

    for i, doc in enumerate(source):
        tmp.add_documents([doc])

        if i % 100000 == 0:
            dictionary.merge_with(tmp)
            tmp = Dictionary()
            print(i, "merged.")

    dictionary.merge_with(tmp)
    dictionary.compactify()
    return dictionary

# Sample sample_n items from the top top_n items in dictionary
def sample_dict(sample_n, top_n, dictionary):
    dictionary.filter_extremes(no_below=0, no_above=1, keep_n=top_n)
    good_ids = random.sample(range(top_n), sample_n)
    dictionary.filter_tokens(good_ids=good_ids)

    return dictionary

def build_model(model_type, save=False, resume=False):
    print(f"Building {model_type} model...")

    if resume:
        done = os.listdir(f"model_{model_type}")
        done = list(filter(lambda x: x.count(".") == 1, done))
        to_build = [lp for lp in dct.values() if lp not in map(lambda x: x.split(".")[0], done)]
    else:
        done = []
        to_build = dct.values()

    for i, lemma_pos in enumerate(to_build, len(done)):
        lemma, pos = lemma_pos.split("_")
        filename = f"Sentences_vertical/{pos}/{lemma}.txt"

        with open(filename, "r", encoding="latin-1") as infile:
            sentences = get_sentences(infile)
            corpus = [dct.doc2bow(doc) for doc in sentences]
            best_umass = -100
            best_model = 0

            for n in range(2, 20):
                if model_type == "hdp":
                    model = HdpModel(corpus=corpus, id2word=dct, T=n, random_state=42)
                elif model_type == "lda":
                    model = LdaModel(corpus=corpus, id2word=dct, num_topics=n, random_state=42)

                cm = CoherenceModel(model=model, corpus=corpus, dictionary=dct, coherence="u_mass")
                cs = cm.get_coherence()

                if cs > best_umass:
                    best_umass = cs
                    best_model = model

            if save == True:
                best_model.save(f"model_{model_type}/{lemma_pos}.sav")

            print(f"{datetime.now().strftime('%H:%M')} {i+1}/{len(dct.values())}")

def _build_base_topic_distribution(model_type, save=False):
    lemma_td = {}

    print(f"Building base topic distribution for {model_type}...")

    for i, lemma_pos in enumerate(dct.values()):
        if model_type == "hdp":
            model = HdpModel.load(f"model_hdp/{lemma_pos}.sav")
        elif model_type == "lda":
            model = LdaModel.load(f"model_lda/{lemma_pos}.sav")

        lemma, pos = lemma_pos.split("_")
        td = defaultdict(int)

        filename = f"Sentences_vertical/{pos}/{lemma}.txt"
        with open(filename, "r", encoding="latin-1") as infile:
            sentences = get_sentences(infile)

            for sentence in sentences:
                topic_prob_dist = model[dct.doc2bow(sentence)]

                if topic_prob_dist != []:
                    best_topic, _ = max(topic_prob_dist, key=lambda t: t[1])
                    td[best_topic] += 1

        print(f"{datetime.now().strftime('%H:%M')} {i+1}/{len(dct)}\t{lemma_pos}")

        lemma_td[lemma_pos] = len(td), td

    if save == True:
        with open(f"results/topic_distribution_{model_type}_None.txt", "w", encoding="utf-8") as outfile:
            for lemma_pos, (n_senses, td) in lemma_td.items():
                outfile.write(f"{lemma_pos}\t{n_senses}\t{dict(td)}\n")

    return lemma_td

def get_topic_distribution(model_type, mode=None, save=False):
    print(f"Building topic distribution for {model_type} ({mode})...")

    lemma_td = {}
    base_td = {}

    if mode is None:
        return _build_base_topic_distribution(model_type, save=save)

    try:
        with open(f"results/topic_distribution_{model_type}_None.txt", "r", encoding="utf-8") as file:
            for line in file:
                lemma_pos, n_senses, td = line.strip().split("\t")
                td = re.sub('(\d+):', r'"\1":', td)
                td = json.loads(td)
                base_td[lemma_pos] = int(n_senses), td

        for lemma_pos, (n_senses, td) in base_td.items():
            if mode == "prune": # Filter topics with < 0.15 prob
                td = dict(filter(lambda x: (x[1]/sum(td.values())) > 0.15, td.items())) 
                n_senses = len(td)
            elif mode == "jdist": # Adjust the number of senses with Jensen-Shannon distance
                p = list(td.values())
                q = [1] * n_senses
                n_senses *= 1 - j_distance(p, q)

            lemma_td[lemma_pos] = n_senses, td

        if save == True:
            with open(f"results/topic_distribution_{model_type}_{mode}.txt", "w", encoding="utf-8") as outfile:
                for lemma_pos, (n_senses, td) in lemma_td.items():
                    outfile.write(f"{lemma_pos}\t{n_senses}\t{dict(td)}\n")
    except FileNotFoundError:
        print("Error: first build base distribution")

    return lemma_td

def get_correlation(model_type, mode, save=False):
    with open("stats/adj.txt", "r", encoding="utf-8") as adj, open("stats/adv.txt", "r", encoding="utf-8") as adv, open("stats/noun.txt", "r", encoding="utf-8") as noun,\
        open("stats/verb.txt", "r", encoding="utf-8") as ver, open(f"results/topic_distribution_{model_type}_{mode}.txt", "r", encoding="utf-8") as lemma_topic:
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

        if save == True:
            with open(f"results/corr_{model_type}_{mode}.txt", "w", encoding="utf-8") as corr:
                corr.write("POS\tWordnet-Correlation\tWordnet-pvalue"
                    "\tWiktionary-Correlation\tWiktionary-pvalue\n")
                for pos, (coeff_wordnet, coeff_wiktionary) in result.items():
                    corr.write(f"{pos}\t{coeff_wordnet.correlation}\t{coeff_wordnet.pvalue}"
                    f"\t{coeff_wiktionary.correlation}\t{coeff_wiktionary.pvalue}\n")

        return result

def do_all():
    filenames = [filename for filename in os.listdir("ITWAC") if filename.startswith("ITWAC-")]
    source = itertools.chain(*[open(f"ITWAC/{filename}", "r", encoding="latin-1") for filename in filenames])
    source = extract_sentences(source)

    sample_sentences(5000, source, save=True)

    for model in ["hdp", "lda"]:
        build_model(model, save=True)

        for mode in [None, "prune", "jdist"]:
            get_topic_distribution(model, mode, save=True)
            get_correlation(model, mode, save=True)

if __name__=="__main__":
    filenames = [filename for filename in os.listdir("ITWAC") if filename.startswith("ITWAC-")]
    source = itertools.chain(*[open(f"ITWAC/{filename}", "r", encoding="latin-1") for filename in filenames])
    #source = open("ITWAC/ITWAC-21.xml", "r", encoding="latin-1")
    source = extract_sentences(source)
    
    sample_sentences(5000, source, save=True)

    model = "hdp"
    build_model(model, save=True)

    for mode in [None, "prune", "jdist"]:
        get_topic_distribution(model, mode, save=True)
        get_correlation(model, mode, save=True)