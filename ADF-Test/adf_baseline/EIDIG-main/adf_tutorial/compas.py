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
import time
from adf_model.tutorial_models import dnn
from adf_utils.utils_tf import model_prediction, model_argmax , layer_out
from adf_tutorial.utils import cluster, gradient_graph
#from IPython.display import clear_output
import numpy as np
import experiments
from preprocessing import pre_census
from preprocessing import pre_compas
from preprocessing import pre_default
from preprocessing import pre_heart
from preprocessing import pre_diabetes
from preprocessing import pre_students
from preprocessing import pre_credit
from preprocessing import pre_bank
# from preprocessing import pre_german_credit
# from preprocessing import pre_bank_marketing
from tensorflow import keras
from EIDIG import individual_discrimination_generation
import os
from tensorflow import keras
import numpy as np
import generation_utilities
import time
import ADF
import EIDIG

dataset={'census':pre_census,'compas':pre_compas,'bank':pre_bank,'heart':pre_heart,'default':pre_default,
        'credit':pre_credit,'students':pre_students,'diabetes':pre_diabetes}
data_set='compas'
for sens in dataset[data_set].protected_attribs:
    sens_attr=[sens]
    RQ2=[]
    for trail in range(1):
        X=dataset[data_set].X
        Y=dataset[data_set].y
        input_shape=dataset[data_set].input_shape
        nb_classes = 2
        tf.set_random_seed(1234)

        config = tf.ConfigProto(device_count = {'GPU': 0})
        config.allow_soft_placement= True

        sess = tf.Session(config=config)
        z = tf.placeholder(tf.float32, shape=input_shape)
        y = tf.placeholder(tf.float32, shape=(None, nb_classes))
        model = dnn(input_shape, nb_classes)   

        preds = model(z)
        saver = tf.train.Saver()
        model_path='../models/'
        model_path = model_path + data_set + "/test.model"
        saver.restore(sess, model_path)

        # construct the gradient graph
        grad_0 = gradient_graph(z, preds)
        num_experiment_round = 1 # the number of experiment rounds
        g_num = 1000# the number of seeds used in the global generation phase
        l_num = 1000 # the maximum search iteration in the local generation phase
        benchmark='C-a'
        protected_attribs=sens_attr
        # experiments.comparison(ROUND, benchmark, pre_census_income.X_train, protected_attribs, pre_census_income.constraint, adult_model, g_num, l_num)
        constraint=dataset[data_set].constraint
        start_time=time.time()
        decay=0.5
        c_num=4
        max_iter=10
        s_g=1.0
        s_l=1.0
        epsilon_l=1e-6
        fashion='RoundRobin'
        num_ids = np.array([0] * 3)
        time_cost = np.array([0] * 3)
        print(data_set,protected_attribs)
        for i in range(num_experiment_round):
            round_now = i + 1
            print('--- ROUND', round_now, '---')
            if g_num >= len(X):
                seeds = X.copy()
            else:
                clustered_data = generation_utilities.clustering(X, c_num)
                seeds = np.empty(shape=(0, len(X[0])))
                for i in range(g_num):
                    new_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i%c_num, fashion=fashion)
                    seeds = np.append(seeds, [new_seed], axis=0)


            t1 = time.time()
            ids_EIDIG_INF, gen_EIDIG_INF, total_iter_EIDIG_INF, tot_g,g_disc,tot_l,l_disc,t_f_g = EIDIG.individual_discrimination_generation(start_time,sess,preds,z,grad_0,X, seeds, protected_attribs, constraint, model, decay, l_num, l_num+1, max_iter, s_g, s_l, epsilon_l)
            np.save('../results/' + data_set + '/ids_EIDIG_INF_' + str(round_now) + '.npy', ids_EIDIG_INF)
            t2 = time.time()
            print('EIDIG-INF:', 'In', total_iter_EIDIG_INF, 'search iterations', len(gen_EIDIG_INF), 'non-duplicate instances are explored', len(ids_EIDIG_INF), 'of which are discriminatory. Time cost:', t2-t1, 's.')
            num_ids[2] += len(ids_EIDIG_INF)
            time_cost[2] += t2-t1
            
            ccc=ids_EIDIG_INF
            print(len(ids_EIDIG_INF))
            print('global',tot_g,g_disc)
            RQ2.append([len(gen_EIDIG_INF)]+[len(ids_EIDIG_INF)]+[l_disc]+[tot_l]+[t_f_g])
            if not os.path.exists('../results/'):
                os.makedirs('../results/')
            if not os.path.exists('../results/' + data_set + '/'):
                os.makedirs('../results/' + data_set + '/')
            np.save('../results/' + data_set + '/EIDIG_sens_'+str(protected_attribs[0])+'_'+str(trail)+'.npy',RQ2)
            print('local',tot_l,l_disc)
            print('time first',t_f_g)
        sess.close()
        tf.reset_default_graph()
        


