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

from adf_data.bank import bank_data

from adf_model.tutorial_models import dnn
from adf_utils.utils_tf import model_prediction, model_argmax , layer_out
from adf_utils.config import bank
from adf_tutorial.utils import cluster, gradient_graph
#from IPython.display import clear_output



FLAGS = flags.FLAGS

# step size of perturbation
perturbation_size = 1

def check_for_error_condition(conf, sess, x, preds, t, sens_params, input_shape, epsillon , ent_tresh):
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
    entropy1 = sh_entropy(pred, ent_tresh)
    entropy2 = sh_entropy(pred, epsillon)
    partition = clustering(pred,samples, sens_params , epsillon)
    

    return  max(list(partition.keys())[1:]) - min(list(partition.keys())[1:]), \
                                            len(partition)-1, entropy1, entropy2, conf#(len(partition) -1),
    
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

    
#-------------------------------------------
    
def dnn_fair_testing(dataset, sens_params, model_path, cluster_num, 
                     max_global, max_local, max_iter, epsillon, max_k_iter, ent_tresh):
    """
    
    The implementation of DaiKe
    :param dataset: the name of testing dataset
    :param sensitive_param: the index of sensitive feature
    :param model_path: the path of testing model
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :param max_global: the maximum number of samples for global search
    :param max_local: the maximum number of samples for local search
    :param max_iter: the maximum iteration of global perturbation
    """
    data = {"bank":bank_data}
    data_config = {"bank":bank}
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

    # store the result of fairness testing
    sens_list=sens_params
    for sen1 in sens_list:
        sens_params=[sen1]
        global max_k
        global start_time
        global max_k_time
        
        print(dataset, sens_params)
        RQ2_table = []
        RQ1_table = []
        min_entropy = []
        k_mean = []
        sh_entropy_mean = []
        for trial in range(10):
            print('Trial', trial)
            if  sess._closed:
                sess = tf.Session(config=config)
                sess = tf.Session(config=config)
                x = tf.placeholder(tf.float32, shape=input_shape)
                y = tf.placeholder(tf.float32, shape=(None, nb_classes))
                model = dnn(input_shape, nb_classes)   
                preds = model(x)
                saver = tf.train.Saver()
                saver.restore(sess, model_path)
                grad_0 = gradient_graph(x, preds)
            global_inputs = set()
            tot_inputs =set()
            global_inputs_list = []
            local_inputs = set()
            local_inputs_list = []
            total_dis_l = 0
            total_dis_l_2 = 0
            total_dis_g = 0
            total_dis_g_2 = 0
#             time_first_disc = 0
            seed_num = 0
            time1 = time.time()    
            max_k_time = 0
            #-----------------------
            def evaluate_local(inp):

                """
                Evaluate whether the test input after local perturbation is an individual discriminatory instance
                :param inp: test input
                :return: whether it is an individual discriminatory instance
                """ 
                global max_k
                global max_k_time
                global start_time
                result, K , entropy1, entropy2, conf = check_for_error_condition(data_config[dataset], sess, x, preds, inp, 
                                                   sens_params, input_shape, epsillon , ent_tresh)    
                if K > max_k:
                    max_k = K 
                    max_k_time = time.time() - start_time

                dis_sample =copy.deepcopy(inp.astype('int').tolist())   
                for sens in sens_params:
                    dis_sample[sens - 1] = 0    
                if tuple(dis_sample) not in global_inputs and\
                                        tuple(dis_sample) not in local_inputs:

                    local_inputs.add(tuple(dis_sample))
                    local_inputs_list.append(dis_sample + [init_k] + [max_k] + [round(max_k_time,4)] + \
                                                  [seed_num] + [round(entropy1,2)] + [round(entropy2,2)] + [time.time() - start_time])

                return (-1 * result)

            # select the seed input for fairness testing
            inputs = seed_test_input(clusters, min(max_global, len(X)))
            for num in range(len(inputs)):

                #clear_output(wait=True)
                start_time = time.time()
                if time.time()-time1 > 900:
                    break 
                print('Input ',seed_num)
                index = inputs[num]
                sample = X[ index : index + 1]

                # start global perturbation
                for iter in range( max_iter + 1 ):            
                    if time.time()-time1 > 900 :
                        break
                    m_sample = m_instance( np.array(sample) , sens_params, data_config[dataset] )
                    pred = pred_prob( sess, x, preds, m_sample , input_shape )
                    entropy1 = sh_entropy(pred, ent_tresh)
                    entropy2 = sh_entropy(pred, epsillon)
                    clus_dic = clustering( pred, m_sample, sens_params, epsillon )

                    if iter == 0:
                        init_k = len(clus_dic) - 1
                        max_k = init_k
                        max_k_time = round((time.time() - start_time),4)

                    if len(clus_dic) - 1 > max_k:
                        max_k = len(clus_dic) - 1
                        max_k_time = round((time.time() - start_time),4)

                    sample,n_values = global_sample_select( clus_dic, sens_params )
                    dis_sample = sample.copy()
                    for sens in sens_params:
                        dis_sample[0][sens  - 1] = 0


                    if tuple(dis_sample[0].astype('int')) not in global_inputs and\
                                        tuple(dis_sample[0].astype('int')) not in local_inputs:
                        dis_flag = True
                        global_inputs.add(tuple(dis_sample[0].astype('int')))
                        global_inputs_list.append(dis_sample[0].astype('int').tolist())

                    else:
                        dis_flag = False


                    if dis_flag and (len(clus_dic)-1 >= 2):    
                   
                        loc_x,n_values = local_sample_select(clus_dic ,sens_params )                              
                        minimizer = {"method": "L-BFGS-B"}
                        local_perturbation = Local_Perturbation(sess, grad_0, x, n_values, 
                                                                sens_params, input_shape[1], 
                                                                data_config[dataset])               
                        basinhopping(evaluate_local, loc_x, stepsize = 1.0, 
                                     take_step = local_perturbation, minimizer_kwargs = minimizer, 
                                     niter = max_local)


                    if dis_flag :
                        global_inputs_list[-1] +=  [init_k , max_k , round(max_k_time,4) ,
                                                         seed_num, round(entropy1,2) , round(entropy2,2), time.time() - start_time]




                    clus_dic = {}
                    if iter == max_iter:
                        break

    #                 if max_k_g < max_k:
    #                     max_k_g = max_k
    #                     clock = 0
    #                 else:
    #                     clock += 1


    #                 if (clock >= 30) and ((max(list(clus_dic.keys())[1:]) - min(list(clus_dic.keys())[1:]) < 20)):#max_k <= 1/(2*epsillon):
    #                     print('Failed to increase K in 30 consequative runs')
    #                     break


                    #Making up n_sample
                    n_sample = sample.copy()
                    for i in range(len(sens_params)):
                        n_sample[0][sens_params[i] - 1] = n_values[i]                

                    # global perturbation

                    s_grad = sess.run(tf.sign(grad_0), feed_dict = {x: sample})
                    n_grad = sess.run(tf.sign(grad_0), feed_dict = {x: n_sample})

                    # find the feature with same impact
                    if np.zeros(data_config[dataset].params).tolist() == s_grad[0].tolist():
                        g_diff = n_grad[0]
                    elif np.zeros(data_config[dataset].params).tolist() == n_grad[0].tolist():
                        g_diff = s_grad[0]
                    else:
                        g_diff = np.array(s_grad[0] == n_grad[0], dtype = float)                
                    for sens in sens_params:
                        g_diff[sens - 1] = 0 
    #                         g_diff1[sens - 1] = 0


                    cal_grad = s_grad * g_diff
    #                     cal_grad1 = np.array([grad_m_sign[np.random.randint(len(grad_m_sign))] * g_diff1])
    #                     if np.zeros(input_shape[1]).tolist() == cal_grad1.tolist()[0]:
    #                         index = np.random.randint(len(cal_grad1[0]) - 1)
    #                         for i in range(len(sens_params) - 1, -1, -1):
    #                             if index == sens_params[i] - 1 :
    #                                 index = index + 1
    #                         cal_grad1[0][index]  = np.random.choice([1.0, -1.0])

                    if np.zeros(input_shape[1]).tolist() == cal_grad.tolist()[0]:
                        index = np.random.randint(len(cal_grad[0]) - 1)
                        for i in range(len(sens_params) - 1, -1, -1):
                            if index == sens_params[i] - 1 :
                                index = index + 1

                        cal_grad[0][index]  = np.random.choice([1.0, -1.0])

                    sample[0] = clip(sample[0] + perturbation_size * cal_grad[0], data_config[dataset]).astype("int")

                seed_num += 1
            #print('Time to find the 1st discrimination',time_first_disc)    
            print('Total Time =', time.time()-time1)

            # create the folder for storing the fairness testing result
            if not os.path.exists('../results/'):
                os.makedirs('../results/')
            if not os.path.exists('../results/' + dataset + '/'):
                os.makedirs('../results/' + dataset + '/')
            if not os.path.exists('../results/' + dataset + '/OurTool/'):
                os.makedirs('../results/' + dataset + '/OurTool/')
            if not os.path.exists('../results/' + dataset + '/OurTool/RQ1&2/'):
                os.makedirs('../results/' + dataset + '/OurTool/RQ1&2/')
            if not os.path.exists('../results/' + dataset + '/OurTool/RQ3/'):
                os.makedirs('../results/' + dataset + '/OurTool/RQ3/')
            if not os.path.exists('../results/' + dataset + '/OurTool/RQ1&2/' + ''.join(str(i) for i in sens_params)+'_10runs/'):
                os.makedirs('../results/' + dataset + '/OurTool/RQ1&2/' + ''.join(str(i) for i in sens_params)+'_10runs/')

            # storing the fairness testing result
            np.save('../results/' + dataset + '/OurTool/RQ1&2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/global_inputs_'+str(trial)+'.npy', 
                    np.array(global_inputs_list))
            np.save('../results/' + dataset + '/OurTool/RQ1&2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/local_inputs_'+str(trial)+'.npy', 
                    np.array(local_inputs_list))
            total_inputs = np.concatenate((local_inputs_list,global_inputs_list), axis=0)
            np.save('../results/' + dataset + '/OurTool/RQ1&2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/total_inputs_'+str(trial)+'.npy',
                    total_inputs)
            # RQ1 & RQ2

            local_sam = np.array(local_inputs_list).astype('int32')
            global_sam = np.array(global_inputs_list).astype('int32')
            temp_g = np.zeros((1,input_shape[1]))
            # Storing result for RQ1 table


            with open('../results/' + dataset + '/OurTool/RQ1&2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/global_inputs_90_'+str(trial)+'.csv', 'w') as f:
                writer = csv.writer(f)
                for ind in range(len(global_inputs_list)):
                    m_sample = m_instance( np.array([global_inputs_list[ind][:-7]]) , sens_params, data_config[dataset] ) 
                    rows = m_sample.reshape((len(m_sample),input_shape[1]))
                    writer.writerows(np.append(rows,[[global_inputs_list[ind][-1]] for i in range(len(m_sample))],axis=1))

            with open('../results/' + dataset + '/OurTool/RQ1&2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/local_inputs_90_'+str(trial)+'.csv', 'w') as f:
                writer = csv.writer(f)
                for ind in range(len(local_inputs_list)):
                    m_sample = m_instance( np.array([local_inputs_list[ind][:-7]]) , sens_params, data_config[dataset] ) 
                    rows = m_sample.reshape((len(m_sample),input_shape[1]))
                    writer.writerows(np.append(rows,[[local_inputs_list[ind][-1]] for i in range(len(m_sample))],axis=1))

            df_l = pd.read_csv('../results/' + dataset + '/OurTool/RQ1&2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/local_inputs_90_'+str(trial)+'.csv',header=None)  
            df_g = pd.read_csv('../results/' + dataset + '/OurTool/RQ1&2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/global_inputs_90_'+str(trial)+'.csv',header=None)

            df_g['label'] = model_argmax(sess, x, preds, df_g.to_numpy()[:,:input_shape[1]])
            df_l['label'] = model_argmax(sess, x, preds, df_l.to_numpy()[:,:input_shape[1]])

            g_pivot = pd.pivot_table(df_g, values="label", index=list(np.setxor1d(df_g.columns[:-1] ,
                                                                                  np.array(sens_params)-1)), aggfunc=np.sum)
            l_pivot = pd.pivot_table(df_l, values="label", index=list(np.setxor1d(df_g.columns[:-1] ,
                                                                               np.array(sens_params)-1)), aggfunc=np.sum)
            
            g_time = g_pivot.index[np.where((g_pivot['label'] > 0) & (g_pivot['label'] < len(m_sample)))[0]].get_level_values(input_shape[1]).values
            l_time = l_pivot.index[np.where((l_pivot['label'] > 0) & (l_pivot['label'] < len(m_sample)))[0]].get_level_values(input_shape[1]).values
            tot_time = np.sort(np.concatenate((l_time, g_time), axis=0 ))
            
            

            g_dis = (len(m_sample) - g_pivot.loc[(g_pivot['label'] > 0) & \
                                                 (g_pivot['label'] < len(m_sample))]['label'].to_numpy()).sum()
            l_dis = (len(m_sample) - l_pivot.loc[(l_pivot['label'] > 0) & \
                                                 (l_pivot['label'] < len(m_sample))]['label'].to_numpy()).sum()

            
            g_dis_adf = len(g_time)
            l_dis_adf = len(l_time)

            
            global_succ_adf = round((g_dis_adf / len(global_sam)) * 100, 1)
            local_succ_adf  = round((l_dis_adf / len(local_sam)) * 100, 1)
            
            tot_df = pd.DataFrame(total_inputs)
            tot_df = tot_df[[i for i in range(input_shape[1])] + [tot_df.columns[-2]]]
            columns = list(tot_df.columns)
            columns[-1] = 'sh_entropy'
            tot_df.columns = columns
            k = []
            disc = []
            for sample in tot_df.to_numpy()[:,:-1]:             
                m_sample = m_instance( np.array([sample]) , sens_params, data_config[dataset] )
                pred = pred_prob( sess, x, preds, m_sample , input_shape )
                clus_dic = clustering( pred, m_sample, sens_params, epsillon )
                if pred.max() > 0.5 and  pred.min()< 0.5:
                    disc.append(1)
                else:
                    disc.append(0)
                k.append(len(clus_dic) - 1)

            tot_df['k'] = k
            tot_df['disc'] = disc
            tot_dis = tot_df.loc[tot_df['disc'] == 1]     
            tot_dis['min_entropy'] = round(np.log2(tot_dis['k'] ),2)
            min_entropy.append( np.mean(tot_dis['min_entropy']))
            k_mean.append(np.mean(tot_dis['k']))
            sh_entropy_mean.append(np.mean(tot_dis['sh_entropy']))
            tot_dis.to_csv('../results/' + dataset + '/OurTool/RQ3/total_disc_' + str(trial)+'.csv')
        
            # reseting the TF graph for the next round
            sess.close()
            tf.reset_default_graph()

            print('Global disc =', g_dis)
            print('local disc  =',  l_dis)
            print('Total loc samples  = ', len(local_sam)) 
            print('Total glob samples = ', len(global_sam)) 
            print('Total Dis = ',g_dis + l_dis)
            print('Total Dis ADF  = ',g_dis_adf + l_dis_adf)
            print('Total Dis ADF G = ',g_dis_adf)
            print('Total Dis ADF L = ',l_dis_adf)
        
            global_succ = round( g_dis / (len(global_sam) * \
                                 len(m_sample)) * 100,1)
            local_succ = round(l_dis  / (len(local_sam) * \
                                 len(m_sample)) * 100,1)



            df_RQ1 = pd.DataFrame(total_inputs)
            df_RQ1.columns = [i for i in range(input_shape[1])] + ['init_k', 'max_k', 'max_k_time','input_num','entropy1','entropy2', 'sample_time']
            row = [len(total_inputs)] + list(df_RQ1[['init_k', 'max_k', 'max_k_time','entropy1','entropy2']].mean())
            
            RQ1_table.append(row)
            RQ2_table.append([g_dis_adf, l_dis_adf, g_dis_adf + l_dis_adf ,local_succ_adf, tot_time[0], tot_time[999] ])


            print('Local search success rate  = ', local_succ, '%')
            print('Global search success rate = ', global_succ, '%')
            print('Local search ADF success rate  = ', local_succ_adf, '%')
            print('Global search ADF success rate = ', global_succ_adf, '%')

        np.save('../results/' + dataset + '/OurTool/RQ1&2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/QID_RQ2_10runs.npy',
                RQ2_table)
        np.save('../results/' + dataset + '/OurTool/RQ1&2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/QID_RQ1_10runs.npy',
                RQ1_table)
        with open('../results/' + dataset + '/OurTool/RQ1&2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/RQ2_table.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['g_adf_disc', 'l_adf_disc','tot_adf_disc','local_succ_adf','time_to_first','time_to_1000'])
                writer.writerow(np.mean(RQ2_table,axis=0))
                writer.writerow(np.std(RQ2_table,axis=0))

        with open('../results/' + dataset + '/OurTool/RQ1&2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/RQ1_table.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['tot_samples','init_k', 'max_k', 'max_k_time','entropy1','entropy2'])
                writer.writerow(np.mean(RQ1_table,axis=0))
                writer.writerow(np.std(RQ1_table,axis=0))

        with open('../results/' + dataset + '/OurTool/RQ3/entropy.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(['k_mean','min_entropy','sh_entropy'])
            writer.writerow([np.mean(k_mean),np.mean(min_entropy), np.mean(sh_entropy_mean) ])
def main(argv=None):
    dnn_fair_testing(dataset = FLAGS.dataset, 
                     sens_params = FLAGS.sens_params,
                     model_path  = FLAGS.model_path,
                     cluster_num = FLAGS.cluster_num,
                     max_global  = FLAGS.max_global,
                     max_local   = FLAGS.max_local,
                     max_iter    = FLAGS.max_iter,
                     epsillon    = FLAGS.epsillon,
                     max_k_iter  = FLAGS.max_k_iter,
                     ent_tresh   = FLAGS.entropy_tresh)

if __name__ == '__main__':
    flags.DEFINE_string("dataset", "bank", "the name of dataset")
    flags.DEFINE_string('model_path', '../models/', 'the path for testing model')
    flags.DEFINE_integer('cluster_num', 4, 'the number of clusters to form as well as the number of centroids to generate')
    flags.DEFINE_integer('max_global', 1000, 'maximum number of samples for global search')
    flags.DEFINE_integer('max_local', 1000, 'maximum number of samples for local search')
    flags.DEFINE_integer('max_iter', 10, 'maximum iteration of global perturbation')
    flags.DEFINE_list('sens_params', [1], 'sensitive parameters index.1 for age, 9 for gender, 8 for race')
    flags.DEFINE_float('epsillon', 0.025, 'the value of epsillon for partitioning')
    flags.DEFINE_float('max_k_iter', 0.3, 'the percentage of max_iter if K is not increased')
    flags.DEFINE_float('entropy_tresh', 0.001, 'the threshold for calculatig entropy')
    tf.app.run()


