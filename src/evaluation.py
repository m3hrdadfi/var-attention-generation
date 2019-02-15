import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist


def sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None):
    min_len_present = min(len(references), len(hypothesis))

    if min_len_present == 0:
        return 0

    if smoothing_function is None:
        smoothing_function = SmoothingFunction()
        smoothing_function = smoothing_function.method7

    try:
        bleu = corpus_bleu(references, hypothesis, weights, smoothing_function)
        return bleu

    except Exception:
        return 0


def calculate_bleu_scores(references, hypotheses):
    """
    Calculates BLEU 1-4 scores based on NLTK functionality

    Args:
        references: List of reference sentences
        hypotheses: List of generated sentences

    Returns:
        bleu_1, bleu_2, bleu_3, bleu_4: BLEU scores

    """
    bleu_1 = np.round(
        sentence_bleu(references, hypotheses, weights=(1.0, 0., 0., 0.)),
        decimals=2
    )
    bleu_2 = np.round(
        sentence_bleu(references, hypotheses, weights=(0.50, 0.50, 0., 0.)),
        decimals=2
    )
    bleu_3 = np.round(
        sentence_bleu(references, hypotheses, weights=(0.34, 0.33, 0.33, 0.)),
        decimals=2
    )
    bleu_4 = np.round(
        sentence_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)),
        decimals=2
    )

    return bleu_1, bleu_2, bleu_3, bleu_4


def calculate_ngram_diversity(corpus):
    """
    Calculates unigram and bigram diversity

    Args:
        corpus: tokenized list of sentences sampled

    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score

    """
    bigram_finder = BigramCollocationFinder.from_words(corpus)

    if bigram_finder.N == 0:
        return 0, 0

    bi_diversity = len(bigram_finder.ngram_fd) / bigram_finder.N

    dist = FreqDist(corpus)
    uni_diversity = len(dist) / len(corpus)

    return uni_diversity, bi_diversity


def calculate_entropy(corpus):
    """
    Calculates diversity in terms of entropy (using unigram probability)

    Args:
        corpus: tokenized list of sentences sampled

    Returns:
        ent: entropy on the sample sentence list

    """
    fdist = FreqDist(corpus)
    total_len = len(corpus)

    if total_len == 0:
        return 0

    ent = 0
    for k, v in fdist.items():
        p = v / total_len

        ent += -p * np.log(p)

    return ent
