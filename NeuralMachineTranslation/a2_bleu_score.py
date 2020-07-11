# Copyright 2020 University of Toronto, all rights reserved

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp  # exp(x) gives e^x


def grouper(seq, n):
    '''Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''
    ngrams = []
    last_index = len(seq) - n
    for i in range(last_index + 1):
	    # append ith n gram to list
	    ngrams.append(tuple(seq[i: i+n]))
    return ngrams

def n_gram_precision(reference, candidate, n):
    '''Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''
    candidate_grams = grouper(candidate, n)
    reference_grams = grouper(reference, n)

    num_candidate_grams = len(candidate_grams)
    num_candidate_grams_in_atleast_one_reference = 0
    for candidate_gram in candidate_grams:
        if candidate_gram in reference_grams:
            num_candidate_grams_in_atleast_one_reference += 1
    return num_candidate_grams_in_atleast_one_reference/num_candidate_grams if num_candidate_grams != 0 else 0

def brevity_penalty(reference, candidate):
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
 
    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''
    if len(candidate) == 0:
        return 0
    brevity = len(reference) / len(candidate)
    return 1 if brevity < 1 else exp(1 - brevity)

def BLEU_score(reference, candidate, n):
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''
    precision_term = 1
    for i in range(n):
        precision_term = precision_term * n_gram_precision(reference, candidate, i + 1)
    return brevity_penalty(reference, candidate) * (precision_term**(1/n))
