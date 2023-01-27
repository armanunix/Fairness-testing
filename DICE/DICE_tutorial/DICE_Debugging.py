

import numpy as np
from itertools import product, combinations
import tensorflow.compat.v1 as tf 
tf.compat.v1.disable_eager_execution()
import sys, os, shutil
sys.path.append("../")
import copy
import time
import pandas as pd
from scipy import stats
from tensorflow.python.platform import flags
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
from DICE_utils.utils_tf import model_prediction, model_argmax , layer_out, model_eval
from DICE_utils.config import census, credit, bank, compas, default, heart, diabetes, students , meps15, meps16
from IPython.display import clear_output
import csv
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", help='The name of dataset: census, credit, bank, default, meps21 ', required=True)
parser.add_argument("-sensitive_index", help='The index for sensitive features', required=True)
parser.add_argument("-timeout", help='Max. running time', default = 3600, required=False)
parser.add_argument("-num_samples", help='Number of samples to debug', default = 1000, required=False)

args = parser.parse_args()

FLAGS = flags.FLAGS
       
def m_instance( sample, sens_params, conf):
    index = []
    m_sample = []
    for sens in sens_params:
        index.append([i for i in range(conf.input_bounds[sens - 1][0], conf.input_bounds[sens-1][1] + 1)])
      
    for ind in list(product(*index)):     
        temp = sample.copy()
        for i in range(len(sens_params)):
            temp[0][sens_params[i]-1] = ind[i]
        m_sample.append(temp)
    return np.array(m_sample)
    
def clustering(probs,m_sample, sens_params, epsilon=0.025):
    cluster_dic = {}
    cluster_dic['Seed'] = m_sample[0][0]
    bins= np.arange(0, 1, epsilon )
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
        
def neuron_locator(sess, model, samples, layer_number,model_path, input_shape, 
                   nb_classes, dataset, sens_params, update_list  ):
        
        if  sess._closed:
#             config = tf.ConfigProto()
#             config.gpu_options.per_process_gpu_memory_fraction = 0.8
            config = tf.ConfigProto(device_count = {'GPU': 0})
            config.allow_soft_placement= True            
            sess   = tf.Session(config = config)
            x      = tf.placeholder(tf.float32, shape = input_shape)
            y      = tf.placeholder(tf.float32, shape = (None, nb_classes))
            model  = dnn(input_shape, nb_classes)   
            preds  = model(x)
            saver  = tf.train.Saver()
            saver.restore(sess, model_path)
            
        num_layers = len(model.layers)
        feed_dic = {}
        for neuron in range(len(update_list)):           
            for layer in range(0,num_layers - 1,2):
                if layer == 0:
                    l = model.layers[layer].fprop(samples.astype('float32'))
                else:
                    l = model.layers[layer].fprop(r)                   
                if layer + 1 == (layer_number * 2) - 1:
                    indices = []
                    for instance in range(l.shape[0]):                       
                        indices.append([ instance, 0, neuron])       
                    updates = [ update_list[ neuron ] ] * l.shape[0]
                    r = model.layers[layer + 1].fprop(l , indices, updates)
                else:
                    r = model.layers[layer + 1].fprop(l)
            feed_dic[neuron] = r
        all_probs = sess.run(feed_dic)
        out_dic   = {}
        for key in all_probs.keys():

            probs = np.array(all_probs[key]).reshape((np.array(all_probs[key]).shape[0],2))[:,1:].reshape((np.array(all_probs[key]).shape[0]))
            clus  = clustering(probs,samples, sens_params)
            out_dic[key] = [len(clus) - 1 ]
        sess.close()
        tf.reset_default_graph()
        return out_dic
    
def model_acc(sess, model,model_path,input_shape, nb_classes,
              dataset, sens_params,neuron,X,Y,layer_number,num_layers,update_list):
        
        if  sess._closed:
#                 config = tf.ConfigProto()
#                 config.gpu_options.per_process_gpu_memory_fraction = 0.8
                config = tf.ConfigProto(device_count = {'GPU': 0})
                config.allow_soft_placement= True
                sess   = tf.Session(config = config)
                x      = tf.placeholder(tf.float32, shape = input_shape)
                y      = tf.placeholder(tf.float32, shape = (None, nb_classes))
                model  = dnn(input_shape, nb_classes)   
                preds  = model(x)
                saver  = tf.train.Saver()
                saver.restore(sess, model_path)
        feed_dic = {}        
        for layer in range(0,num_layers - 1,2):
            if layer == 0:
                l = model.layers[layer].fprop(X.astype('float32'))
            else:
                l = model.layers[layer].fprop(r)          
            if layer + 1 == (layer_number * 2) - 1:
                indices = []
                for instance in range(l.shape[0]):                       
                    indices.append([ instance, neuron])                
                updates = [ update_list[ neuron ] ] * l.shape[0]                
                r = model.layers[layer + 1].fprop(l , indices, updates)
            else:
                r = model.layers[layer + 1].fprop(l)             
        all_probs = sess.run(r)
        out_class = []
        for out in all_probs:
            out_class.append(np.argmax(out))
        truth_val = []
        for tr in Y:
                truth_val.append(np.argmax(tr))
        acc = 0
        for i in range(len(out_class)):
            if out_class[i] == truth_val[i]:
                acc += 1
        accuracy = round(acc/len(out_class),3)
        sess.close()
        tf.reset_default_graph()
        return accuracy 

def get_rate(sess, model, model_path, input_shape, nb_classes,
              dataset, lay_name, layer_output):
         
        def get_distance(vec1, vec2, size):
            return abs(vec1 - vec2).sum() / size
        
        max_dis = 0
        epsillon = 10 ** -7
        num_samples = len(layer_output[lay_name])
        #print('lay_name',lay_name)
        layer_ind = np.where(np.array(list(layer_output.keys())) == lay_name)[0][0]

        for ind in range(layer_ind):
            temp_dis = 0
            if 'ReLU' in np.array(list(layer_output.keys()))[ind]:
                layer_name = np.array(list(layer_output.keys()))[ind]
                layer_size  = len(layer_output[layer_name][0][0])
                distances = np.zeros((num_samples,num_samples))
                
                for i in combinations(range(num_samples),2):
                    distances[i[0],i[1]] = get_distance(layer_output[layer_name][i[0]],
                                                        layer_output[layer_name][i[1]],layer_size)
                if distances.max()> max_dis:
                    max_dis = distances.max()                                      
        distances = np.zeros((num_samples,num_samples))       
        for i in combinations(range(num_samples),2):
            distances[i[0],i[1]] = get_distance(layer_output[lay_name][i[0]],layer_output[lay_name][i[1]],len(layer_output[lay_name][0][0]))
        cur_dis = distances.max()
        change_rate = (cur_dis - max_dis ) / (max_dis + epsillon)
        return change_rate
    
def layer_locator(sess, model, model_path,sens_params, input_shape, nb_classes,
              dataset,conf, samples):
        if  sess._closed:
    #                 config = tf.ConfigProto()
    #                 config.gpu_options.per_process_gpu_memory_fraction = 0.8
            config = tf.ConfigProto(device_count = {'GPU': 0})
            config.allow_soft_placement= True
            sess   = tf.Session(config = config)
            x      = tf.placeholder(tf.float32, shape = input_shape)
            y      = tf.placeholder(tf.float32, shape = (None, nb_classes))
            model  = dnn(input_shape, nb_classes)   
            preds  = model(x)
            saver  = tf.train.Saver()
            saver.restore(sess, model_path)
            
        layer_list = []
        layer_rate = []
        for sample in samples:            
            samples = m_instance( np.array([sample]) , sens_params, conf)
            layer_output = layer_out(sess,model,np.array(samples).astype('float32')) 
            temp_list = []
            for layer in layer_output.keys():
                if 'ReLU' in layer:
                    temp_rate = get_rate(sess, model, model_path, input_shape, nb_classes,
                                          dataset,layer, layer_output)
                    temp_list.append(temp_rate) 
            layer_rate.append(max(temp_list[1:]))
                              
            layer_list.append((np.argmax(np.array(temp_list[1:])) + 2 ))
        sess.close()
        tf.reset_default_graph()
        print()
        return stats.mode(layer_list)[0][0], np.array(layer_rate).mean()    
#-------------------------------------------
    
def dnn_fair_testing(dataset, sens_params, model_path, acc_e, timeout, num_samples):

    data = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas":compas_data, 
            "default": default_data, "heart":heart_data, "diabetes":diabetes_data, 
            "students":students_data, "meps15":meps15_data, "meps16":meps16_data}
    data_config = {"census":census, "credit":credit, "bank":bank, "compas":compas, "default":default,
                  "heart":heart , "diabetes":diabetes,"students":students, "meps15":meps15, "meps16":meps16}
    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    tf.set_random_seed(1234)
    layer_numbers=[]
    layer_influence = []
    RQ3_table = []
    for trial in range(1):
        config = tf.ConfigProto(device_count = {'GPU': 0})
        config.allow_soft_placement= True
        sess  = tf.Session(config = config)
        x     = tf.placeholder(tf.float32, shape = input_shape)
        y     = tf.placeholder(tf.float32, shape = (None, nb_classes))
        model = dnn(input_shape, nb_classes)   
        preds = model(x)
        saver = tf.train.Saver()
        model_path ='../models/'
        model_path = model_path + dataset + "/test.model"
        saver.restore(sess, model_path)
        eval_params = {'batch_size': 128}
        ini_acc = round(model_eval(sess, x, y, preds, X, Y, args=eval_params),3)
        time1 = time.time()
    # Loading the result of QID
        layer_output = layer_out(sess,model,X.astype('float32'))
        input_df  = pd.read_csv('../results/' + str(dataset) + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/total_disc_'+str(trial)+'.csv',header='infer')
        input_df = input_df.drop(columns=['Unnamed: 0'])
        if 'time' not in input_df.columns:
            input_df['time'] = 0
        sample_df = input_df.copy()
        sample_df_rand = sample_df.sample(n = int(num_samples*0.8),axis = 0,random_state = np.random.RandomState())
        sample_df_maxk = sample_df.sort_values(by = 'k',ascending=False).head(int(num_samples*0.2))
        sample_df = pd.concat([sample_df_maxk,sample_df_rand])
        ini_k_samples = sample_df['k']
        sample_df = sample_df.drop(columns = ['sh_entropy', 'k', 'disc', 'min_entropy','time']) 
        samples   = sample_df.to_numpy()
        num_samples = len(samples)
        print(' Localizing biased layer')
        layer_number, layer_rate = layer_locator(sess, model, model_path, sens_params, input_shape, nb_classes,
              dataset,data_config[dataset], samples)

        layer_numbers.append(layer_number)
        layer_influence.append(layer_rate)
        print(' Localizing biased neuron')
        #-----------------------------
        update_df = layer_output['ReLU'+str((2*layer_number) - 1 )]
        update_min  = np.min(update_df,axis=0)
#         update_max  = np.max(update_df,axis=0)
        update_mean = np.mean(update_df,axis=0)
#         update_std  = np.std(update_df,axis=0)
        update_list = []
        update_list.append(update_min)      
        update_list.append(update_mean)
        layer_size   = model.layers[(layer_number*2) - 1].input_shape[1]
        layer_name   = model.layers[(layer_number*2) - 1]
        num_layers   = len(model.layers)
        num_trial = len(update_list)
        all_dic = {}
        accu_neuron = {}
        acc_try = {}
        sample_ind = 0
        for sample in samples:
            if time.time() - time1 > timeout:
                break
            update_list_man = np.array([0] * layer_size)
            m_samples  = m_instance( np.array([sample]), sens_params, data_config[dataset])
            change_dic = {}
            for i in range(num_trial):
                update_list_man = update_list[i]
                x = neuron_locator(sess, model, m_samples, layer_number,model_path,
                               input_shape, nb_classes, dataset, sens_params, update_list_man )
                if sample_ind == 0:
                    accu_neuron = {}
                    for neuron in range(len(update_list_man)):
                        accu_neuron[neuron] = model_acc(sess, model,model_path,
                                         input_shape, nb_classes, dataset, sens_params,
                                         neuron,X,Y,layer_number,num_layers,update_list_man)
                    acc_try[i] = accu_neuron                 
                change_dic[i] = x  
            all_dic[sample_ind] = change_dic
            clear_output(wait=False)
            sample_ind += 1

        # create the folder for storing the fairness testing result
        if not os.path.exists('../results/'):
            os.makedirs('../results/')
        if not os.path.exists('../results/' + str(dataset) + '/'):
            os.makedirs('../results/' + str(dataset) + '/')
        if not os.path.exists('../results/' + str(dataset) + '/DICE/'):
            os.makedirs('../results/' + str(dataset) + '/DICE/')
        if not os.path.exists('../results/' + str(dataset) + '/DICE/RQ3/'):
            os.makedirs('../results/' + str(dataset) + '/DICE/RQ3/')   

        accu_dic =  acc_try 
        ini_k = np.array(ini_k_samples[:sample_ind])
        print('Tested samples ', sample_ind)
        num_samples = len(all_dic.keys())
        num_force   = len(all_dic[0].keys())
        num_neuron  = len(all_dic[0][0].keys())
        print('Biased layer',layer_number)
        ini_k = np.repeat(ini_k, (num_force * num_neuron))
        data  = np.zeros(((num_samples * num_force * num_neuron) ,4) , dtype = 'int32')
        df    = pd.DataFrame(data,columns = ['sample','force','neuron','K'],dtype = 'int32')
        sample_col = np.repeat(np.array([i for i in range(num_samples)]),(num_neuron * num_force))
        force_col  = np.array([ int(i/num_neuron ) for i in range( num_neuron * num_force ) ] * num_samples)
        neuron_col = np.array([i for i in range( num_neuron )] * ( num_samples*num_force ))
        df['sample'] = sample_col
        df['force']  = force_col
        df['neuron'] = neuron_col
        df['acc'] = 0
        acc = pd.DataFrame(accu_dic).transpose().to_numpy()
        acc = acc.reshape(acc.shape[0] * acc.shape[1],)
        for i in range(len(all_dic.keys())):
            temp = pd.DataFrame(all_dic[i]).transpose().to_numpy()
            temp = temp.reshape(((len(all_dic[0][0].keys())) * len(all_dic[0].keys()),))   
            df.loc[df.loc[(df['sample'] == i) ].index,'acc'] = acc
            df.loc[df.loc[(df['sample'] == i) ].index,'K'] = temp
        df['K'] = df['K'].transform(lambda x:x[0])
        df['init_k'] = ini_k

        R_act   = []
        R_deact = []
        diff_R  = []
        #acc_e   = acc_epsilon
        for neuron in range(num_neuron):
            k_deact = df.loc[(df['neuron'] == neuron) & (df['force'] == 0) & \
                                  (df['acc'] >= ini_acc - acc_e)]['K'].mean()
            k_act   = df.loc[(df['neuron'] == neuron) & (df['force'] == 1) & \
                                  (df['acc'] >= ini_acc - acc_e)]['K'].mean()
            k_init  = df.loc[(df['neuron'] == neuron) & (df['acc'] >= ini_acc - acc_e)]['init_k'].mean()
            R_act_temp   = (k_act - k_init) / k_init
            R_deact_temp = (k_deact - k_init) / k_init
            diff_R_temp  = R_act_temp - R_deact_temp
            R_act.append(R_act_temp)
            R_deact.append(R_deact_temp)
            diff_R.append(diff_R_temp)

        RQ3_table.append(diff_R)
  
    RQ3_table = np.mean(RQ3_table,axis=0)
    pos_eff_ind = np.where(RQ3_table<0)[0]
    neg_eff_ind = np.where(RQ3_table>0)[0]
    neg_neurons = neg_eff_ind[np.argsort(RQ3_table[neg_eff_ind])]
    pos_neurons = pos_eff_ind[np.argsort(RQ3_table[pos_eff_ind])]
    
    if len(neg_neurons) > 0:
        N_neg_1 = neg_neurons[-1]
        ACD_neg_1 = RQ3_table[N_neg_1]
    else:
        N_neg_1 = 'N/A'
        ACD_neg_1 = 'N/A'
    if len(neg_neurons) > 1:
        N_neg_2 = neg_neurons[-2]
        ACD_neg_2 = RQ3_table[N_neg_2]
    else:
        N_neg_2 = 'N/A'
        ACD_neg_2 = 'N/A'
    if len(neg_neurons) > 2:
        N_neg_3 = neg_neurons[-3]
        ACD_neg_3 = RQ3_table[N_neg_3]
    else:
        N_neg_3 = 'N/A'
        ACD_neg_3 = 'N/A'

    if len(pos_neurons) > 0:
        N_pos_1 = pos_neurons[0]
        ACD_pos_1 = RQ3_table[N_pos_1]
    else:
        N_pos_1 = 'N/A'
        ACD_pos_1 = 'N/A'

    if len(pos_neurons) > 1:
        N_pos_2 = pos_neurons[1]
        ACD_pos_2 = RQ3_table[N_pos_2]
    else:
        N_pos_2 = 'N/A'
        ACD_pos_2 = 'N/A'
    if len(pos_neurons) > 2:
        N_pos_3 = pos_neurons[2]
        ACD_pos_3 = RQ3_table[N_pos_3]
    else:
        N_pos_3 = 'N/A'
        ACD_pos_3 = 'N/A'


    with open('../results/'+str(dataset)+'/DICE/RQ3_table_1_'+str(num_samples)+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Biased layer'] + [' Layer influence'] + ['Neuron+1','ACD+1', 'Neuron+2','ACD+2',
                                                                   'Neuron+3', 'ACD+3','Neuron-1', 'ACD-1',
                                                                   'Neuron-2','ACD-2','Neuron-3','ACD-3'])

        writer.writerow(([stats.mode(layer_numbers)[0][0],np.mean(layer_influence),N_pos_1,ACD_pos_1,
              N_pos_2,ACD_pos_2, N_pos_3,ACD_pos_3, N_neg_1,ACD_neg_1,N_neg_2,ACD_neg_2,
              N_neg_3,ACD_neg_3]))

    shutil.move('../results/'+str(dataset)+'/DICE/RQ3_table_1_'+str(num_samples)+'.csv', '../results/'+str(dataset)+'/DICE/RQ3/RQ3_table_1_'+str(num_samples)+'.csv')
    A          = ini_acc 
    K_ini      = df.groupby('sample').mean()['init_k'].mean()
    A_deactive = df.loc[(df['neuron']==N_neg_1) & (df['force']==0)]['acc'].mean()
    K_deactive = df.loc[(df['neuron']==N_neg_1) & (df['force']==0)]['K'].mean()
    A_active   = df.loc[(df['neuron']==N_pos_1) & (df['force']==1)]['acc'].mean()
    K_active   = df.loc[(df['neuron']==N_pos_1) & (df['force']==1)]['K'].mean()
    
    with open('../results/'+str(dataset)+'/DICE/RQ3/RQ3_table_2_'+str(num_samples)+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['A', 'K', 'A=0', 'K=0', 'A>0', 'K>0' ,'time to debug'])
        writer.writerow([ A, K_ini, A_deactive, K_deactive, A_active, K_active , round(time.time() - time1,1)])
    print('Time to intervene', time.time() - time1 )
def main(argv = None):
    
    dnn_fair_testing(dataset = FLAGS.dataset, 
                     sens_params = FLAGS.sens_params,
                     model_path  = FLAGS.model_path,
                     acc_e = FLAGS.acc_epsilon, 
                     timeout = FLAGS.timeout,
                     num_samples = FLAGS.num_samples
                    )


if __name__ == '__main__':
    sens_list = [int(i) for i in re.findall('[0-9]+', args.sensitive_index)]
    flags.DEFINE_string("dataset", args.dataset, "the name of dataset")
    flags.DEFINE_string('model_path', '../models/', 'the path for testing model')
    flags.DEFINE_list('sens_params',sens_list,'sensitive parameters index.1 for age, 9 for gender, 8 for race')
    flags.DEFINE_float('acc_epsilon',0.05,'Tolerance epsilon')
    flags.DEFINE_integer('timeout', args.timeout, 'search timeout')
    flags.DEFINE_integer('num_samples', args.num_samples, 'Number of samples for debugging')
    tf.app.run()

