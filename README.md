# Automatic induction of the number of word senses with Topic Modeling techniques

This project was developed as a thesis project whose aim is to use topic modeling techniques to induce the number of word senses. The topics induced by the topic models are used as a proxy for the senses.

Topic Modeling is a technique to find clusters in a corpus based on the co-occurrence of words. Each cluster represents a "topic" in the corpus, that is, a collection of words that refer to the same subject matter. Therefore, a topic model on a corpus reflects the topics covered in the corpus.

The aim of this project is to exploit the clustering capabilities of topic modeling to find the number of senses for a Word Sense Induction (WSI) task. The core idea is that different meanings of the same word occur in different contexts and, therefore, along different words. The topic model is able to capture these differences and cluster the documents in the corpus accordingly, providing a correspondence between the clusters and the senses of the word.

In order to evaluate the senses induced by the topic model, the induced senses are compared against two human-annotated repositories: WordNet and Wiktionary.

# Methodology
## Preprocessing
The corpus on which the analysis is done is itWaC. For each token, the corpus provides its lemma and part-of-speech (POS) tag. The corpus is preprocessed so that only the lemma and POS tag are considered for each word. Specifically, the lemmas considered are those that belong to four POS: adjectives, adverbs, nouns and verbs.

## Dictionary
After preprocessing the corpus, a Gensim dictionary is built on it. Then, the dictionary is filtered so that only lemmas reported in both Wordnet and Wiktionary remain. Finally, the lemmas on which the analysis is done are the most frequent 5000 lemmas in the dictionary.

## Sampling
For each lemma, 5000 sentences that contain it are randomly sampled.

## Topic Modeling
After sampling the sentences, a topic model is trained for each lemma, where the corpus for the topic model is the set of sampled sentences for that lemma.

The two topic models tested are Hierarchical Dirichlet Process (HDP) and Latent Dirichlet Allocation (LDA). HDP induces automatically the number of senses, whereas in LDA it needs to be specified beforehand. Thus, to determine the best number of topics using LDA it is necessary to train multiple models, each with a different number of topics, and then select the best among those using the coherence score metric (the chosen coherence measure is u_mass).

## Topic distribution
After obtaining the topic models for each lemma, their topic distribution are calculated. The result of a topic model inference on a document is a probability distribution over the topics, meaning that each topic is assigned a value based on the probability of the document being accurately represented by that topic.

Finally, all these topic distributions are summed to  to obtain the final topic distribution for that lemma over all the corpus. The number of induced senses was compared to the number of senses reported in WordNet and Wiktionary.

## Topic distribution alterations
The base topic distribution, described above, or the number of topics were altered in various ways to perform experiments.

### Jensen-Shannon distance
The Jensen-Shannon distance measures how much two distributions p and q differ from each other. In this case, it was used to measure how much does the topic distribution differ from a uniform distribution and alter the number of senses accordingly. The more the topic distribution is equal to a uniform distribution, the more the number of induced topics is reduced.

### Low-probability topic pruning
The topics with low probability are removed from the distribution. In this case, the cutoff probability has been chosen to be 0.15.

### Pareto principle
The Pareto principle states that, in many real-world scenarios, roughly, "20% of causes account for 80% of outcomes". In this case, this is interpreted as the top 20% of most frequent senses account for over 80% of occurrences. This assumption makes it so that curve of the distribution looks like (unsurprisingly) a discrete Pareto distribution. Thus, the number of senses is cut so that only the most frequent senses that sum to 0.80 are left. Since it is almost impossible for the sum to be exactly equal to 0.80 in a real-world scenario, the requirement of equality has been relaxed; instead, the method finds the index which minimizes the difference between the sum and 0.80. 

### Equilibrium index
This method is based on the same concept as the Pareto principle method, that is, the most frequent senses account for the majority of occurrences. This method tries to find the index in the topic distribution (sorted by probability, descending) where the sum of the probabilities on the left of the index is equal to the sum of the probabilities on the right of the index. Since it is almost impossible for these two sums to be equal in a real-world scenario, the require- ment of equality has been relaxed; instead, the method finds the index which minimizes the absolute difference between the two sums.

### Elbow method
This method is also based on the concept that the most frequent senses account for the majority of occurrences. Considering that the distribution forms a curve similar to a Pareto distribution, a method to find the number of the most relevant topics is to find the "elbow" of the curve.

### Kneedle
This method is simply another way to calculate the elbow described above. This method is the Kneedle method.

# Results
The results are considered relevant when the correlation's absolute value is greater than 0.2 and the p-value is smaller than 0.05. The relevant results are as follows:

|Model|Topic distribution|Correlation|p-value|Ontology|Part of speech|
|-----|------|------------------|-----------|-------|---|
|HDP|Jensen-Shannon|0.3029|0.0017|Wordnet|ADV
|HDP|Equilibrium index|0.2612|0.0074|Wordnet|ADV
|HDP|Pareto principle|0.3149|0.0011|Wordnet|ADV
|LDA|Pareto principle|0.2215|0.0231|Wordnet|ADV