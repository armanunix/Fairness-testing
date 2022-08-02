import numpy as np
import sys
sys.path.append("../")

def arash_data():
    """
    Prepare the data of dataset Default Credit
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("../datasets/arsh", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [i for i in line1[:-2]]
            X.append(L)
            H = [i for i in line1[-2:]]
            Y.append(H)
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 10)
    nb_classes = 2

    return X, Y, input_shape, nb_classes
