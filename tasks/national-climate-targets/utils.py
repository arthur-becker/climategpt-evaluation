import datasets
import numpy as np
from enum import Enum
import itertools

class Label(Enum):
    REDUCTION = "Reduction"
    NET_ZERO = "Net zero"
    OTHER = "Other"


# PRE-PROCESSING

def power_set(iterable):
    "power_set([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))

def _get_few_shots(dataset : datasets.Dataset):
    """
    The dataset is very imbalanced, and the first several examples represent the same
    classes which makes the standard few-shot implementation of LM Evaluation Harness
    unsuitable. Therefore, this function implements the search for own few-shot examples
    with the following requirements:
    1. There have to be exactly 8 examples covering each class in the multi-label
        setting with 3 binary labels.
    2. For each class, a second example is also drawn in order to be used when the first example is used
        as the main question.

    Args:
    `doc` - document with the actual question

    Return:
    `few_shots : dict` - a dictionary of docs as few-shots
    """
    NUM_EXAMPLES_PER_CLASS = 2

    data_classes = power_set([Label.REDUCTION, Label.NET_ZERO, Label.OTHER])
    few_shots_dict = {}
    for labels in data_classes:
        few_shots_dict[labels] = []

    for item in dataset:
        for labels in few_shots_dict:
            if len(few_shots_dict[labels]) < NUM_EXAMPLES_PER_CLASS:
                reduction = int(Label.REDUCTION in labels)
                nzt = int(Label.NET_ZERO in labels)
                other = int(Label.OTHER in labels)

                class_match = reduction == item['annotation_Reduction'] \
                                    and nzt == item["annotation_NZT"] \
                                    and other == item["annotation_Other"]
                if class_match:
                    few_shots_dict[labels].append(item)
        
        if all([len(value) == NUM_EXAMPLES_PER_CLASS for value in few_shots_dict.values()]):
            break

    assert all([len(value) == NUM_EXAMPLES_PER_CLASS for value in few_shots_dict.values()])

    return few_shots_dict

def _get_label(doc, target_type: Label):
    label = None
    if target_type == Label.REDUCTION:
        label = int(doc["annotation_Reduction"])
    elif target_type == Label.NET_ZERO:
        label = int(doc["annotation_NZT"])
    elif target_type == Label.OTHER:
        label = int(doc["annotation_Other"])
    else:
        raise ValueError("Unknown target_type")
    
    return label

def _question_template(doc, target_type: Label, show_answer=False):
    label = _get_label(doc, target_type)
    return f"""
\n\n\n#########################################\n\n\n
The given paragraph:\n{doc["text"]}\n\n
Should the given paragraph be classified as \"Net zero\"?
A. No
B. Yes
\n\nAnswer:{" " + ["A","B"][label] if show_answer else ""}"""

def _process_docs(dataset: datasets.Dataset, target_type: Label) -> datasets.Dataset:
    few_shots = _get_few_shots(dataset)

    def _process_doc(doc):
        # Concatenate few-shot examples and the question
        question_text = ""
        for doc_list in few_shots.values():
            example = doc_list[0] if not (doc_list[0] == doc) else doc_list[1]
            question_text += _question_template(example, target_type, show_answer=True)
        question_text += _question_template(doc, target_type, show_answer=False)

        # Add new fields
        output = {
            'question_text': question_text,
            'label': _get_label(doc, target_type)
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