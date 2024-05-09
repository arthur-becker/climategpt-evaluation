import datasets
import numpy as np
from enum import Enum

class Label(Enum):
    REDUCTION = "Reduction"
    NET_ZERO = "Net zero"
    OTHER = "Other"

_CHOICES = {
  'A': [], # 0
  'B' : [Label.REDUCTION], # 1
  'C' : [Label.NET_ZERO], # 2
  'D' : [Label.OTHER], # 3
  'E' : [Label.REDUCTION, Label.NET_ZERO], # 4
  'F' : [Label.REDUCTION, Label.OTHER], # 5
  'G' : [Label.NET_ZERO, Label.OTHER], # 6
  'H' : [Label.REDUCTION, Label.NET_ZERO, Label.OTHER] # 7
}



# PRE-PROCESSING

def _get_choices():
    return list(_CHOICES.keys())

def _get_enumerated_choices():
    text = ""

    assert len(_CHOICES.keys()) > 0

    for key in _CHOICES.keys():
        line = f"{key}. "
        labels : list[Label] = _CHOICES[key]
        if len(labels) == 0:
            line += "(none)"
        else:
            for i in range(len(labels)):
                label = labels[i]
                line += label.value
                if i + 1 < len(labels):
                    line += ", "
        text += line + "\n"
    return text

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        output = {
            'choices': _get_choices(),
            'enumerated_choices': _get_enumerated_choices()
        }
        return output

    return dataset.map(_process_doc)



# POST-PROCESSING

def process_results_mcq(doc, results, true_choices, is_ref_positive = None):
    results = [result[0] for result in results]

    chosen = int(np.argmax(results))
    acc = 1.0 if chosen in true_choices else 0.0

    ref = 1 if is_ref_positive else 0
    pred = None
    if (chosen in true_choices) and is_ref_positive:
        pred = 1
    elif (chosen not in true_choices) and (not is_ref_positive):
        pred = 1
    else:
        pred = 0

    if is_ref_positive is None:
        return {
            "acc": acc
        }
    else:
        return {
            "acc": acc,
            "precision": (pred, ref),
            "recall": (pred, ref),
            "f1": (pred, ref)
        }

def _get_positive_choices_by_label(label: Label) -> list[int]:
    """
    Returns indices of all positive choices for a given label assuming a single binary classifier. For instance, given `label=Label.REDUCTION`,
    this function returns all indices of choices where the label `reduction` is meant to be positive (`Reduction`, `Reduction, Net Zero` etc.)
    """
    result = []
    choices = list(_CHOICES.keys())
    for i in range(len(choices)):
        choice = choices[i]
        choice_labels = _CHOICES[choice]
        if label in choice_labels:
            result.append(i)

    if len(result) == 0:
        raise ValueError(f"There are no choices assuming positive `label={label.name}`")
    return result

def _get_negative_choices_by_label(label: Label) -> list[int]:
    all_labels = list(range(len(_CHOICES)))
    positive_labels = _get_positive_choices_by_label(label)
    return [x for x in all_labels if x not in positive_labels]


def _labels_to_choice(label_list: list[Label]) -> int:
    label_list = label_list
    choices = list(_CHOICES.keys())
    for i in range(len(choices)):
        choice = choices[i]
        choice_labels = _CHOICES[choice]
        if set(choice_labels) == set(label_list):
            return i 
    raise ValueError(f"Unknown combination of labels: {label_list}")

def evaluate_multilabel(doc, results):
    labels = []
    if doc["annotation_Reduction"]:
        labels.append(Label.REDUCTION)
    if doc["annotation_NZT"]:
        labels.append(Label.NET_ZERO)
    if doc["annotation_Other"]:
        labels.append(Label.OTHER)

    choice_index = _labels_to_choice(labels)
    return process_results_mcq(doc, results, [choice_index])

def evaluate_reduction_label(doc, results):
    true_choices = None
    ref = False
    if doc["annotation_Reduction"]:
        ref = True
        true_choices = _get_positive_choices_by_label(Label.REDUCTION)
    else:
        true_choices = _get_negative_choices_by_label(Label.REDUCTION)
    return process_results_mcq(doc, results, true_choices, is_ref_positive=ref)

def evaluate_nz_label(doc, results):
    true_choices = None
    ref = False
    if doc["annotation_NZT"]:
        ref = True
        true_choices = _get_positive_choices_by_label(Label.NET_ZERO)
    else:
        true_choices = _get_negative_choices_by_label(Label.NET_ZERO)
    return process_results_mcq(doc, results, true_choices, is_ref_positive=ref)

def evaluate_other_label(doc, results):
    true_choices = None
    ref = False
    if doc["annotation_Other"]:
        ref = True
        true_choices = _get_positive_choices_by_label(Label.OTHER)
    else:
        true_choices = _get_negative_choices_by_label(Label.OTHER)
    return process_results_mcq(doc, results, true_choices, is_ref_positive=ref)

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
        return 1.0 * precision * recall / (precision + recall)