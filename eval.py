#!/usr/bin/env python
# coding=utf-8

import argparse
import re
from collections import Counter

from nlgeval import NLGEval
from nltk import ngrams
from nltk.tokenize import word_tokenize


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    s = s.lower()
    re_art = re.compile(r'\b(a|an|the)\b')
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = s.replace('... ', '')
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s


def postprocess_text(preds, labels):
    preds = [pred.lower().strip() for pred in preds]
    labels = [label.lower().strip() for label in labels]

    preds = [' '.join(word_tokenize(pred)) for pred in preds]
    labels = [' '.join(word_tokenize(label)) for label in labels]

    return preds, labels


def compute_distinct(preds):
    unigram_counter, bigram_counter = Counter([]), Counter([])
    for pred in preds:
        pred_for_cal = pred.split()
        unigram_counter.update(pred_for_cal)
        bigram_counter.update(ngrams(pred_for_cal, 2))

    try:
        distinct_1 = len(unigram_counter) / (sum(unigram_counter.values()))
    except ZeroDivisionError:
        distinct_1 = 0
    try:
        distinct_2 = len(bigram_counter) / (sum(bigram_counter.values()))
    except ZeroDivisionError:
        distinct_2 = 0

    return distinct_1, distinct_2


def evaluate(hyp_list, ref_list):
    # Metric
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)

    distinct_1, distinct_2 = compute_distinct([normalize_answer(s) for s in hyp_list])
    hyp_list, ref_list = postprocess_text(hyp_list, ref_list)
    result = nlgeval.compute_metrics([ref_list], hyp_list)
    result['distinct_1'] = distinct_1
    result['distinct_2'] = distinct_2
    result = {k: round(v * 100, 4) for k, v in result.items()}

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hypothesis",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--references",
        type=str,
        default=None,
        required=True,
    )
    args = parser.parse_args()
    with open(args.hypothesis) as f:
        hyp_list = f.readlines()
    with open(args.references) as f:
        ref_list = f.readlines()

    indices_to_delete = []
    for i, l in enumerate(hyp_list):
        l = l.lower()
        if "i'm sorry" in l or "response" in l:
            indices_to_delete.append(i)

    hyp_list = [hyp_list[i] for i in range(len(hyp_list)) if i not in indices_to_delete]
    ref_list = [ref_list[i] for i in range(len(ref_list)) if i not in indices_to_delete]

    # Metric
    nlgeval = NLGEval(no_skipthoughts=True, metrics_to_omit=["SPICE"])

    distinct_1, distinct_2 = compute_distinct([normalize_answer(s) for s in hyp_list])
    # distinct_1, distinct_2 = compute_distinct([s for s in hyp_list])
    hyp_list, ref_list = postprocess_text(hyp_list, ref_list)
    result = nlgeval.compute_metrics([ref_list], hyp_list)
    result['distinct_1'] = distinct_1
    result['distinct_2'] = distinct_2
    result = {k: round(v * 100, 4) for k, v in result.items()}
    print(result)
