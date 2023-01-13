"""
This python file preprocesses the Census Income Dataset.
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

"""
    https://www.kaggle.com/vivamoto/us-adult-income-update?select=census.csv
"""

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
data_path = ('../datasets/census.csv')
df = pd.read_csv(data_path)


data=df.to_numpy()
# split data into training data, validation data and test data
X = data[:, :-1]
y = data[:, -1]
X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)


# set constraints for each attribute, 117936000 data points in the input space
constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T
input_shape=(None,len(X[0]))

# for census income data, age(0), race(6) and gender(7) are protected attributes in 12 features
protected_attribs = [0, 7, 8]