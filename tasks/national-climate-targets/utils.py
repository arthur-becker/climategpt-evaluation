import datasets
import numpy as np
from enum import Enum

class Label(Enum):
    REDUCTION = 0
    NET_ZERO = 1
    OTHER = 2

# TODO: replace choices with single letter
_CHOICES = {
  'None': [], # 0
  'Reduction' : [Label.REDUCTION], # 1
  'Net zero' : [Label.NET_ZERO], # 2
  'Other' : [Label.OTHER], # 3
  'Reduction, Net zero' : [Label.REDUCTION, Label.NET_ZERO], # 4
  'Reduction, Other' : [Label.REDUCTION, Label.OTHER], # 5
  'Net zero, Other' : [Label.NET_ZERO, Label.OTHER], # 6
  'Reduction, Net zero, Other' : [Label.REDUCTION, Label.NET_ZERO, Label.OTHER] # 7
}



# PRE-PROCESSING

def _get_choices():
    return list(_CHOICES.keys())

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        output = {
            'choices': _get_choices(),
        }
        return output

    return dataset.map(_process_doc)



# POST-PROCESSING

def process_results_mcq(doc, results, true_labels):
    results = [result[0] for result in results]

    acc = 1.0 if int(np.argmax(results)) in true_labels else 0.0
    completion_len = np.array([float(len(i)) for i in doc["choices"]])
    acc_norm = 1.0 if int(np.argmax(results / completion_len)) in true_labels else 0.0

    return {
        "acc": acc,
        "acc_norm": acc_norm,
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
    positive_choices = _get_positive_choices_by_label(Label.REDUCTION)
    return process_results_mcq(doc, results, positive_choices)

def evaluate_nz_label(doc, results):
    positive_choices = _get_positive_choices_by_label(Label.NET_ZERO)
    return process_results_mcq(doc, results, positive_choices)

def evaluate_other_label(doc, results):
    positive_choices = _get_positive_choices_by_label(Label.OTHER)
    return process_results_mcq(doc, results, positive_choices)