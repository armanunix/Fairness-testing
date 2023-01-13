"""
This python file preprocesses the Census Income Dataset.
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from aif360.datasets.meps_dataset_panel19_fy2015 import MEPSDataset19
cd = MEPSDataset19()
le = LabelEncoder()
df = pd.DataFrame(cd.features)
df[0] = pd.cut(df[0],9, labels=[i for i in range(1,10)])
df[2] = pd.cut(df[0],10, labels=[i for i in range(1,11)])
df[3] = pd.cut(df[0],10, labels=[i for i in range(1,11)])
df = df.astype('int').drop(columns=[10])
df[4] = le.fit_transform(df[4])


# make outputs stable across runs
np.random.seed(42)
tf.random.set_seed(42)


def set_table(vocab):
    # set lookup table for categorical attributes
    indices = tf.range(len(vocab), dtype=tf.int64)
    table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
    num_oov_buckets = 1
    table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
    return table


# load adult dataset, and eliminate unneccessary features


data=df.to_numpy()
# split data into training data, validation data and test data
X = np.array(df.to_numpy(), dtype=int)
y = np.array(cd.labels, dtype=int)
y = np.eye(2)[y.reshape(-1)]
y = np.array(y, dtype=int)
# X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)


# set constraints for each attribute, 117936000 data points in the input space
constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T
input_shape=(None,len(X[0]))

# for census income data, age(0), race(6) and gender(7) are protected attributes in 12 features
protected_attribs = [1, 2, 9]