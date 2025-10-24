import numpy as np
# softmax
from scipy.special import softmax
def normalize_logits(row, tau=15):
    logits = np.array([float(x) for x in row['logits'].split(',')])
    normalized = softmax(logits / tau)
    return ','.join(map(str, normalized))