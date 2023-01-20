import sys
sys.path.append("../")
from sklearn.cluster import KMeans
#####from sklearn.externals import joblib
import joblib
sys.modules['sklearn.externals.joblib'] = joblib


import os
##import tensorflow as tf
import tensorflow.compat.v1 as tf 
from tensorflow.python.platform import flags

from DICE_data.census import census_data
from DICE_data.credit import credit_data
from DICE_data.bank import bank_data
from DICE_data.compas import compas_data
from DICE_data.default import default_data
from DICE_data.heart import heart_data
from DICE_data.diabetes import diabetes_data
from DICE_data.students import students_data
from DICE_data.meps15 import meps15_data
from DICE_data.meps16 import meps16_data

from DICE_utils.utils_tf import model_loss

FLAGS = flags.FLAGS

datasets_dict = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas":compas_data, 
            "default": default_data, "heart":heart_data, "diabetes":diabetes_data, 
            "students":students_data, "meps15":meps15_data, "meps16":meps16_data}

def cluster(dataset, cluster_num=4):
    """
    Construct the K-means clustering model to increase the complexity of discrimination
    :param dataset: the name of dataset
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :return: the K_means clustering model
    """
    if os.path.exists('../clusters/' + dataset + '.pkl'):
        clf = joblib.load('../clusters/' + dataset + '.pkl')
    else:
        X, Y, input_shape, nb_classes = datasets_dict[dataset]()
        clf = KMeans(n_clusters=cluster_num, random_state=2019).fit(X)
        joblib.dump(clf , '../clusters/' + dataset + '.pkl')
    return clf
    
    

def gradient_graph(x, preds, y=None):
    """
    Construct the TF graph of gradient
    :param x: the input placeholder
    :param preds: the model's symbolic output
    :return: the gradient graph
    """
    if y == None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keepdims=True)
#       # Tensor flow 2 uses cast instead of to_float 
        y = tf.equal(preds, preds_max)       
        y = tf.cast(y, tf.float32)       
        #y = tf.to_float(tf.equal(preds, preds_max))        
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keepdims=True)

    # Compute loss
    loss = model_loss(y, preds, mean=False)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)
    #print(grad.numpy())
    return grad

def main(argv=None):
    cluster(dataset=FLAGS.dataset,
            cluster_num=FLAGS.clusters)

if __name__ == '__main__':
    flags.DEFINE_string('dataset', 'census', 'name of datasets')
    flags.DEFINE_integer('clusters', 4, 'number of clusters')

    tf.app.run()
