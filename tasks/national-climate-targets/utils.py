import datasets
import numpy as np

# TODO: replace choices with single letter
_CHOICES = [
  'None', # 0
  'Reduction', # 1
  'Net zero', # 2
  'Other', # 3
  'Reduction, Net zero', # 4
  'Reduction, Other', # 5
  'Net zero, Other', # 6
  'Reduction, Net zero, Other' # 7
]

def _extract_labels(doc):
    reduction = doc["annotation_Reduction"]
    net_zero = doc["annotation_NZT"]
    other = doc["annotation_Other"]
    return reduction, net_zero, other

def _labels_to_choice(reduction, net_zero, other) -> int:
    assert reduction in [0,1]
    assert net_zero in [0,1]
    assert other in [0,1]

    label_str = None
    if reduction == 1:
        if net_zero == 1:
            if other == 1:
                # 'Reduction, Net zero, Other'
                return 7
            else:
                # 'Reduction, Net zero'
                return 4
        else:
            if other == 1:
                # 'Reduction, Other'
                return 5
            else:
                # 'Reduction'
                return 1
    else:
        if net_zero == 1:
            if other == 1:
                # 'Net zero, Other'
                return 6
            else:
                # 'Net zero'
                return 2
        else:
            if other == 1:
                # 'Other'
                return 3
            else:
                # 'None'
                return 0

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        output = {
            'choices': _CHOICES
        }
        return output

    return dataset.map(_process_doc)

def process_results_mcq(doc, results, true_labels):
    results = [result[0] for result in results]

    acc = 1.0 if int(np.argmax(results)) in true_labels else 0.0
    completion_len = np.array([float(len(i)) for i in doc["choices"]])
    acc_norm = 1.0 if int(np.argmax(results / completion_len)) in true_labels else 0.0

    return {
        "acc": acc,
        "acc_norm": acc_norm,
    }

def process_results_total(doc, results):
    reduction, net_zero, other = _extract_labels(doc)
    choice_index = _labels_to_choice(reduction, net_zero, other)
    return process_results_mcq(doc, results, [choice_index])