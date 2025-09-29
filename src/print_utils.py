import numpy as np

CLEAR_TERMINAL = "\033[H\033[J"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
EOC = "\033[0m"
GRAY = "\033[90m"

def pretty_vec(x,p=2):
    return np.array2string(np.array(x), precision=p, floatmode='fixed')