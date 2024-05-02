import datasets

_DOC_TO_CHOICE = [
  'None', # 0
  'Reduction', # 1
  'Net zero', # 2
  'Other', # 3
  'Reduction, Net zero', # 4
  'Reduction, Other', # 5
  'Net zero, Other', # 6
  'Reduction, Net zero, Other' # 7
]

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
        reduction = doc["annotation_Reduction"]
        net_zero = doc["annotation_NZT"]
        other = doc["annotation_Other"]

        output = {
            'label': _labels_to_choice(reduction, net_zero, other)
        }
        return output

    return dataset.map(_process_doc)