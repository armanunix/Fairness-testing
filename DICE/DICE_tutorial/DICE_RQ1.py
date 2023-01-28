import numpy as np
import csv
from itertools import product
import itertools
import tensorflow.compat.v1 as tf 
tf.compat.v1.disable_eager_execution()
import sys, os
sys.path.append("../")
import copy
import pandas as pd
from tensorflow.python.platform import flags
from scipy.optimize import basinhopping
from scipy.stats import entropy
import time

from DICE_data.census import census_data
from DICE_data.credit import credit_data
from DICE_data.compas import compas_data
from DICE_data.default import default_data
from DICE_data.bank import bank_data
from DICE_data.heart import heart_data
from DICE_data.diabetes import diabetes_data
from DICE_data.students import students_data
from DICE_data.meps15 import meps15_data
from DICE_data.meps16 import meps16_data

from DICE_model.tutorial_models import dnn
from DICE_utils.utils_tf import model_prediction, model_argmax , layer_out
from DICE_utils.config import census, credit, bank, compas, default, heart, diabetes, students , meps15, meps16
from DICE_tutorial.utils import cluster, gradient_graph
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", help='The name of dataset: census, credit, bank, default, meps21 ', required=True)
parser.add_argument("-sensitive_index", help='The index for sensitive features', required=True)
args = parser.parse_args()

FLAGS = flags.FLAGS

# step size of perturbation
perturbation_size = 1

def check_for_error_condition(conf, sess, x, preds, t, sens_params, input_shape, epsillon):
    """
    Check whether the test case is an individual discriminatory instance
    :param conf: the configuration of dataset
    :param sess: TF session
    :param x: input placeholder
    :param preds: the model's symbolic output
    :param t: test case
    :param sens: the index of sensitive feature
    :return: whether it is an individual discriminatory instance
    """

    t = [t.astype('int')]   
    samples = m_instance( np.array(t), sens_params, conf ) 
    pred = pred_prob(sess, x, preds, samples , input_shape )
    partition = clustering(pred,samples, sens_params , epsillon)
    #entropy_min = np.log2(len(partition)-1)#sh_entropy(pred, ent_tresh)
    #entropy_sh = sh_entropy(pred, epsillon)
    

    return  max(list(partition.keys())[1:]) - min(list(partition.keys())[1:]), \
                                            len(partition)-1, conf#(len(partition) -1),
    
def seed_test_input(clusters, limit):
    """
    Select the seed inputs for fairness testing
    :param clusters: the results of K-means clustering
    :param limit: the size of seed inputs wanted
    :return: a sequence of seed inputs
    """
    i = 0
    rows = []
    max_size = max([len(c[0]) for c in clusters])
    while i < max_size:
        if len(rows) == limit:
            break
        for c in clusters:
            if i >= len(c[0]):
                continue
            row = c[0][i]
            rows.append(row)
            if len(rows) == limit:
                break
        i += 1
    return np.array(rows)

def clip(input, conf):
    """
    Clip the generating instance with each feature to make sure it is valid
    :param input: generating instance
    :param conf: the configuration of dataset
    :return: a valid generating instance
    """
    for i in range(len(input)):
        input[i] = max(input[i], conf.input_bounds[i][0])
        input[i] = min(input[i], conf.input_bounds[i][1])
    return input

class Local_Perturbation(object):
    """
    The  implementation of local perturbation
    """

    def __init__(self, sess, grad, x, n_values, sens_params, input_shape, conf):
        """
        Initial function of local perturbation
        :param sess: TF session
        :param grad: the gradient graph
        :param x: input placeholder
        :param n_value: the discriminatory value of sensitive feature
        :param sens_param: the index of sensitive feature
        :param input_shape: the shape of dataset
        :param conf: the configuration of dataset
        """
        self.sess = sess
        self.grad = grad
        self.x = x
        self.n_values = n_values
        self.input_shape = input_shape
        self.sens = sens_params
        self.conf = conf

    def __call__(self, x):
        """
        Local perturbation
        :param x: input instance for local perturbation
        :return: new potential individual discriminatory instance
        """

        # perturbation
        s = np.random.choice([1.0, -1.0]) * perturbation_size

        n_x = x.copy()
        for i in range(len(self.sens)):
            n_x[self.sens[i] - 1] = self.n_values[i]
       
        # compute the gradients of an individual discriminatory instance pairs
        ind_grad = self.sess.run(self.grad, feed_dict={self.x:np.array([x])})
        n_ind_grad = self.sess.run(self.grad, feed_dict={self.x:np.array([n_x])})

        if np.zeros(self.input_shape).tolist() == ind_grad[0].tolist() and \
           np.zeros(self.input_shape).tolist() == n_ind_grad[0].tolist():
            
            probs = 1.0 / (self.input_shape) * np.ones(self.input_shape)

            for sens in self.sens :
                probs[sens - 1] = 0

                

        else:
            # nomalize the reciprocal of gradients (prefer the low impactful feature)
            grad_sum = 1.0 / (abs(ind_grad[0]) + abs(n_ind_grad[0]))

            for sens in self.sens :
                grad_sum[ sens - 1 ] = 0

            probs = grad_sum / np.sum(grad_sum)
        probs = probs / probs.sum()
        if True in np.isnan(probs):
            probs = 1.0 / (self.input_shape) * np.ones(self.input_shape)

            for sens in self.sens :
                probs[sens - 1] = 0
            probs = probs/probs.sum()


        # randomly choose the feature for local perturbation
        index = np.random.choice(range(self.input_shape) , p=probs)
        local_cal_grad = np.zeros(self.input_shape)
        local_cal_grad[index] = 1.0
        x = clip(x + s * local_cal_grad, self.conf).astype("int")
        return x
                
#--------------------------------------
def m_instance( sample, sens_params, conf):
    index = []
    m_sample = []
    for sens in sens_params:
        index.append([i for i in range(conf.input_bounds[sens - 1][0], conf.input_bounds[sens - 1][1] + 1)])
      
    for ind in list(product(*index)):     
        temp = sample.copy()
        for i in range(len(sens_params)):
            temp[0][sens_params[i] - 1] = ind[i]
        m_sample.append(temp)
    return np.array(m_sample)

def global_sample_select(clus_dic, sens_params):
    leng = 0
    for key in clus_dic.keys():
        if key == 'Seed':
            continue
        if len(clus_dic[key]) > leng:
            leng = len(clus_dic[key])
            largest = key
    
    sample_ind = np.random.randint(len(clus_dic[largest]))
    n_sample_ind = np.random.randint(len(clus_dic[largest]))
    
    sample = clus_dic['Seed']
    for i in range(len(sens_params)):
        sample[sens_params[i] -1] = clus_dic[largest][sample_ind][i]
    # returns one sample of largest partition and its pair
    return np.array([sample]),clus_dic[largest][n_sample_ind]


def local_sample_select(clus_dic, sens_params):
      
    k_1 = min(list(clus_dic.keys())[1:])
    k_2 = max(list(clus_dic.keys())[1:])
    
    sample_ind = np.random.randint(len(clus_dic[k_1]))
    n_sample_ind = np.random.randint(len(clus_dic[k_2]))

    sample = clus_dic['Seed']
    for i in range(len(sens_params)):
        sample[sens_params[i] -1] = clus_dic[k_1][sample_ind][i]
    return np.array([sample]),clus_dic[k_2][n_sample_ind]
    

def clustering(probs,m_sample, sens_params, epsillon):
    cluster_dic = {}
    cluster_dic['Seed'] = m_sample[0][0]
    bins= np.arange(0, 1, epsillon )
    digitized = np.digitize(probs, bins) - 1
    for  k in range(len(digitized)):

        if digitized[k] not in cluster_dic.keys():        
            cluster_dic[digitized[k]]=[ [m_sample[k][0][j - 1] for j in sens_params]]
        else:
            cluster_dic[digitized[k]].append( [m_sample[k][0][j - 1] for j in sens_params])
    return cluster_dic 
    
def pred_prob(sess, x, preds, m_sample, input_shape):
        probs = model_prediction(sess, x, preds, np.array(m_sample).reshape(len(m_sample),
                                                                input_shape[1]))[:,1:2].reshape(len(m_sample))
        return probs

def sh_entropy(probs,bin_thresh, base=2 ):
    bins = np.arange(0, 1,bin_thresh )
    digitized = np.digitize(probs, bins)
    value,counts = np.unique(digitized, return_counts=True)   
    return entropy(counts, base=base)
     
    
def dnn_fair_testing(dataset, sens_params, model_path, cluster_num, 
                     max_global, max_local, max_iter, epsillon):
    """
    
    The implementation of ADF
    :param dataset: the name of testing dataset
    :param sensitive_param: the index of sensitive feature
    :param model_path: the path of testing model
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :param max_global: the maximum number of samples for global search
    :param max_local: the maximum number of samples for local search
    :param max_iter: the maximum iteration of global perturbation
    """
    data = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas":compas_data, 
            "default": default_data, "heart":heart_data, "diabetes":diabetes_data, 
            "students":students_data, "meps15":meps15_data, "meps16":meps16_data}
    data_config = {"census":census, "credit":credit, "bank":bank, "compas":compas, "default":default,
                  "heart":heart , "diabetes":diabetes,"students":students, "meps15":meps15, "meps16":meps16}
    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    tf.set_random_seed(1234)

    config = tf.ConfigProto(device_count = {'GPU': 0})
    config.allow_soft_placement= True

    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)   

    preds = model(x)
    saver = tf.train.Saver()
    model_path = model_path + dataset + "/test.model"
    saver.restore(sess, model_path)

    # construct the gradient graph
    grad_0 = gradient_graph(x, preds)

    # build the clustering model
    clf = cluster(dataset, cluster_num)
    clusters = [np.where(clf.labels_ == i) for i in range(cluster_num)]
    len_msamples = 1
    for sens in sens_params:
        len_msamples *= (data_config[dataset].input_bounds[sens - 1][1] - data_config[dataset].input_bounds[sens - 1][0] + 1) 
    m_sample=[i for i in range(len_msamples)]
    # store the result of fairness testing
    init_k_list =[]
    max_k_list = []
    max_k_time_list = []
    RQ1_table =[]
    for trial in range(1):
        df_l = pd.read_csv('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/local_inputs_90_'+str(trial)+'.csv',header=None)  
        df_g = pd.read_csv('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/global_inputs_90_'+str(trial)+'.csv',header=None)
        total_inputs = np.load('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/total_inputs_'+str(trial)+'.npy')
        init_k_list.append(np.mean(total_inputs[:,input_shape[1]]))
        max_k_list.append(np.mean(total_inputs[:,input_shape[1]+1]))
        max_k_time_list.append(np.mean(total_inputs[:,input_shape[1]+2]))
        df_g['label'] = model_argmax(sess, x, preds, df_g.to_numpy()[:,:input_shape[1]])
        df_l['label'] = model_argmax(sess, x, preds, df_l.to_numpy()[:,:input_shape[1]])
        g_pivot = pd.pivot_table(df_g, values="label", index=list(np.setxor1d(df_g.columns[:-1] ,
                                                                              np.array(sens_params)-1)), aggfunc=np.sum)
        l_pivot = pd.pivot_table(df_l, values="label", index=list(np.setxor1d(df_l.columns[:-1] ,
                                                                           np.array(sens_params)-1)), aggfunc=np.sum)


        g_dis = (len(m_sample) - g_pivot.loc[(g_pivot['label'] > 0) & \
                                             (g_pivot['label'] < len(m_sample))]['label'].to_numpy()).sum()
        l_dis = (len(m_sample) - l_pivot.loc[(l_pivot['label'] > 0) & \
                                             (l_pivot['label'] < len(m_sample))]['label'].to_numpy()).sum()

        tot_df = pd.DataFrame(total_inputs[:,:input_shape[1]])
        
        tot_df.columns=[i for i in range(input_shape[1])]
        

        k = []
        disc = []
        tot_df['sh_entropy'] = 0
        for sam_ind in range(total_inputs.shape[0]): 

            m_sample = m_instance( np.array([total_inputs[:,:input_shape[1]][sam_ind]]) , sens_params, data_config[dataset] )
            pred = pred_prob( sess, x, preds, m_sample , input_shape )
            clus_dic = clustering( pred, m_sample, sens_params, epsillon )
            tot_df.loc[[sam_ind], 'sh_entropy'] = sh_entropy(pred, epsillon)
            if pred.max() > 0.5 and  pred.min()< 0.5:
                disc.append(1)
            else:
                disc.append(0)
            k.append(len(clus_dic) - 1)

        tot_df['k'] = k
        tot_df['disc'] = disc
        tot_df['min_entropy'] = round(np.log2(tot_df['k'] ),2)
        tot_dis = tot_df.loc[tot_df['disc'] == 1]
        tot_dis.to_csv('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/total_disc_' + str(trial)+'.csv')

        haighest_k = np.sort(tot_df['k'].unique())[-3:]
        if len(haighest_k)>2:
            IK1F = np.where(tot_df['k']==haighest_k[2])[0].shape[0]
            IK2F = np.where(tot_df['k']==haighest_k[1])[0].shape[0]
            IK3F = np.where(tot_df['k']==haighest_k[0])[0].shape[0]

        else:
            IK1F = np.where(tot_df['k']==haighest_k[1])[0].shape[0]
            IK2F = np.where(tot_df['k']==haighest_k[0])[0].shape[0]
            IK3F = 0
        print('Run ',trial)
        

        row = [len(total_inputs)] + [np.mean(init_k_list), np.mean(max_k_list),np.mean(max_k_time_list)] + list(tot_dis[['min_entropy', 'sh_entropy']].mean())  + [IK1F, IK2F,IK3F] 
        RQ1_table.append(row)


        
    

    np.save('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/QID_RQ1_10runs.npy',
            RQ1_table)

    with open('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/RQ1_table.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['#I','K_I', 'K_F', 'T_KF','Q_inf',
                         'Q_1', 'IK1F', 'IK2F', 'IK3F'])
            writer.writerow(np.mean(RQ1_table,axis=0))
            writer.writerow(np.std(RQ1_table,axis=0))
    
    
                
def main(argv=None):
    dnn_fair_testing(dataset = FLAGS.dataset, 
                     sens_params = FLAGS.sens_params,
                     model_path  = FLAGS.model_path,
                     cluster_num = FLAGS.cluster_num,
                     max_global  = FLAGS.max_global,
                     max_local   = FLAGS.max_local,
                     max_iter    = FLAGS.max_iter,
                     epsillon    = FLAGS.epsillon)
    
if __name__ == '__main__':    
    sens_list = [int(i) for i in re.findall('[0-9]+', args.sensitive_index)]
    flags.DEFINE_string("dataset",args.dataset, "the name of dataset")
    flags.DEFINE_string('model_path', '../models/', 'the path for testing model')
    flags.DEFINE_integer('cluster_num', 4, 'the number of clusters to form as well as the number of centroids to generate')
    flags.DEFINE_integer('max_global', 1000, 'maximum number of samples for global search')
    flags.DEFINE_integer('max_local', 1000, 'maximum number of samples for local search')
    flags.DEFINE_integer('max_iter', 10, 'maximum iteration of global perturbation')
    
    #if result for RQ1 table: set this to [9,8,1], if result for RQ2 table: set this one sensitive attribute each time 
    # e.g. for census dataset, set [9], [8], [1] for sex, race and age respectively
    flags.DEFINE_list('sens_params', sens_list, 'sensitive parameters index.1 for age, 9 for gender, 8 for race')
    flags.DEFINE_float('epsillon', 0.025, 'the value of epsillon for partitioning')
    tf.app.run()


