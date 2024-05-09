import datasets
import numpy as np
from enum import Enum

class Label(Enum):
    REDUCTION = "Reduction"
    NET_ZERO = "Net zero"
    OTHER = "Other"


# PRE-PROCESSING

def _process_docs(dataset: datasets.Dataset, target_type: Label) -> datasets.Dataset:
    def _process_doc(doc):
        label = None
        if target_type == Label.REDUCTION:
            label = int(doc["annotation_Reduction"])
        elif target_type == Label.NET_ZERO:
            label = int(doc["annotation_NZT"])
        elif target_type == Label.OTHER:
            label = int(doc["annotation_Other"])
        else:
            raise ValueError("Unknown target_type")

        output = {
            'label': label
        }
        return output

    return dataset.map(_process_doc)

def process_docs_reduction(dataset: datasets.Dataset) -> datasets.Dataset:
    return _process_docs(dataset, Label.REDUCTION)

def process_docs_nz(dataset: datasets.Dataset) -> datasets.Dataset:
    return _process_docs(dataset, Label.NET_ZERO)

def process_docs_other(dataset: datasets.Dataset) -> datasets.Dataset:
    return _process_docs(dataset, Label.OTHER)


# POST-PROCESSING

def process_results_mcq(doc, results):
    results = [result[0] for result in results]

    pred = int(np.argmax(results))
    ref = doc["label"]

    acc = 1.0 if pred == ref else 0.0

    return {
        "acc": acc,
        "precision": (pred, ref),
        "recall": (pred, ref),
        "f1": (pred, ref)
    }


# METRICS

def get_confusion_matrix(items):
    true_negatives = sum([int(x == (0,0)) for x in items])
    true_positives = sum([int(x == (1,1)) for x in items])
    false_positives = sum([int(x == (1,0)) for x in items])
    false_negatives = sum([int(x == (0,1)) for x in items])

    return true_positives, true_negatives, false_positives, false_negatives


def precision_aggr(items):
    tp, tn, fp, fn = get_confusion_matrix(items)
    if tp + fp == 0:
        return -1
    else:
        return 1.0 * tp / (tp + fp)

def recall_aggr(items):
    tp, tn, fp, fn = get_confusion_matrix(items)
    if tp + fn == 0:
        return -1
    else:
        return 1.0 * tp / (tp + fn)

def f1_aggr(items):
    precision = precision_aggr(items)
    recall = recall_aggr(items)
    if precision + recall == 0:
        return -1
    else:
        return 2.0 * precision * recall / (precision + recall)