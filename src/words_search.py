import json

from nltk.corpus import wordnet as wn

from config import LANGUAGE, PATH_DICTIONARIES


def _search_wordnet(pos: str):
    """
    Searches Wordnet for lemmas having the specified part-of-speech tag.

    Parameters
    ----------
    pos: str
        The part of speech tag of which to get senses from Wordnet. The possible values are:
            "adj": adjectives
            "adv": adverbs
            "noun": nouns
            "ver": verbs

    Returns
    -------
    dict[lemma, int]
        A dictionary with lemmas as keys and their number of senses as values, sorted by value.
    """
    match LANGUAGE:
        case "en":
            lang = "eng"
        case "it":
            lang = "ita"

    pos_translation = {
        "adj": wn.ADJ,
        "adv": wn.ADV,
        "noun": wn.NOUN,
        "ver": wn.VERB
    }

    pos = pos_translation[pos]
    synset_list = wn.all_synsets(lang=lang)

    wordnet_dict = {}

    for synset in synset_list:
        lemma_list = synset.lemma_names(lang=lang)

        for lemma in lemma_list:
            sense_number = len(wn.synsets(lemma, lang=lang, pos=pos))

            if sense_number > 0:
                lemma = lemma.replace("_", " ").lower()
                wordnet_dict[lemma] = sense_number

    if "gap!" in wordnet_dict:
        del wordnet_dict["gap!"]
    if "pseudogap!" in wordnet_dict:
        del wordnet_dict["pseudogap!"]

    wordnet_dict = dict(sorted(wordnet_dict.items(), key=lambda x: x[1], reverse=True))
    return wordnet_dict

def _search_wiktionary(pos: str):
    """
    Searches Wiktionary for polysemous words (words with more than one sense).

    Parameters
    ----------
    pos: str
        The part of speech tag of which to get senses from Wordnet. The possible values are:
            "adj": adjectives
            "adv": adverbs
            "noun": nouns
            "ver": verbs

    Returns
    -------
    dict[lemma, int]
        A dictionary with lemmas as keys and their number of senses as values, sorted by value.
    """
    wiktionary_dict = {}

    with open(f"{PATH_DICTIONARIES}/kaikki_{pos}.txt", 'r', encoding="utf-8") as infile:
        synset_list = [json.loads(line.strip()) for line in infile.readlines()]

    for synset in synset_list:
        senses = synset["senses"]
        sense_number = len(senses)

        if sense_number > 0:
            lemma = synset["word"]
            lemma = lemma.lower()
            wiktionary_dict[lemma] = sense_number

    wiktionary_dict = dict(sorted(wiktionary_dict.items(), key=lambda x: x[1], reverse=True))

    return wiktionary_dict

def build_senses():
    """
    Builds list of lemma_pos paired with its senses in both Wordnet and Wiktionary.
    Only single words are included, i.e. words with spaces and dashes are excluded.
    """
    print("Building senses...")
    with open(f"{PATH_DICTIONARIES}/dct_onto.tsv", "w", encoding="utf-8") as outfile:
        for pos in ["adj", "adv", "noun", "ver"]:
            wnet = _search_wordnet(pos)
            wikt = _search_wiktionary(pos)

            intersection = wnet.keys() & wikt.keys()
            intersection = set(i for i in intersection if " " not in i and "-" not in i)

            for lemma in intersection:
                n_wnet = wnet[lemma]
                n_wikt = wikt[lemma]

                outfile.write(f"{lemma}_{pos.upper()}\t{n_wnet}\t{n_wikt}\n")

    print("Built senses.")
