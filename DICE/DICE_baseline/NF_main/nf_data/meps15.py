import numpy as np
import pandas as pd
import sys
sys.path.append("../")
from sklearn.preprocessing import LabelEncoder
from aif360.datasets.meps_dataset_panel19_fy2015 import MEPSDataset19
cd = MEPSDataset19()
le = LabelEncoder()
df = pd.DataFrame(cd.features)
df[0] = pd.cut(df[0],9, labels=[i for i in range(1,10)])
df[2] = pd.cut(df[2],10, labels=[i for i in range(1,11)])
df[3] = pd.cut(df[3],10, labels=[i for i in range(1,11)])
df = df.astype('int').drop(columns=[10])
df[4] = le.fit_transform(df[4])

def meps15_data():
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = np.array(df.to_numpy(), dtype=int)
    Y = np.array(cd.labels, dtype=int)
    Y = np.eye(2)[Y.reshape(-1)]
    Y = np.array(Y, dtype=int)
    input_shape = (None, len(X[0]))
    nb_classes = 2
    return X, Y, input_shape, nb_classes
