import numpy as np

def label2onehot(label, class_num):
    result = []
    label = label.reshape([1, -1]).squeeze()
    for v in label:
        encoding = np.zeros([1, class_num])
        encoding[0, int(v)] = 1
        result.append(encoding)

    return np.array(result).squeeze()
