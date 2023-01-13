
import  os
#import psutil
# p = psutil.Process(os.getpid())
# p.cpu_affinity(0)
import numpy as np
from itertools import product, combinations
import tensorflow.compat.v1 as tf 
tf.compat.v1.disable_eager_execution()
import sys
sys.path.append("../")
import copy
import time
import pandas as pd
from scipy import stats
from tensorflow.python.platform import flags
from adf_data.compas import compas_data
from adf_model.tutorial_models import dnn
from adf_utils.utils_tf import model_prediction, model_argmax , layer_out, model_eval
from adf_utils.config import compas
from IPython.display import clear_output


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
    
def clustering(probs,m_sample, sens_params):
    epsillon = 0.025
    cluster_dic = {}
    cluster_dic['Seed'] = m_sample[0][0]
         
    for i in range(len(probs)):
        #  to avoid k = Max + 1
        if probs[i] == 1.0:
            if (int( probs[i] / epsillon ) -1) not in cluster_dic.keys():
             
                cluster_dic[ (int( probs[i] / epsillon ) - 1)] = [ [m_sample[i][0][j - 1] for j in sens_params] ]
           
            else:
                cluster_dic[ (int( probs[i] / epsillon ) - 1)].append( [m_sample[i][0][j - 1] for j in sens_params] )

                       
        elif int( probs[i] / epsillon ) not in cluster_dic.keys():
                cluster_dic[ int( probs[i] / epsillon )] = [ [m_sample[i][0][j - 1] for j in sens_params] ]
           
        else:
                cluster_dic[ int( probs[i] / epsillon)].append( [m_sample[i][0][j - 1] for j in sens_params] )

    return cluster_dic  

    
def pred_prob(sess, x, preds, m_sample, input_shape):
        probs = model_prediction(sess, x, preds, np.array(m_sample).reshape(len(m_sample),
                                    input_shape[1]))[:,1:2].reshape(len(m_sample))
        return probs        
        
def neuron_locator(sess, model, samples, layer_number,model_path, input_shape, 
                   nb_classes, dataset, sens_params, update_list ):
        
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
            probs = np.array(all_probs[key]).reshape((12,2))[:,1:].reshape((12))
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
        for sample in samples:            
            samples = m_instance( np.array([sample]) , sens_params, conf)
            layer_output = layer_out(sess,model,np.array(samples).astype('float32')) 
            temp_list = []
            for layer in layer_output.keys():
                if 'ReLU' in layer:
                    temp_rate = get_rate(sess, model, model_path, input_shape, nb_classes,
                                          dataset,layer, layer_output)
                    temp_list.append(temp_rate)             
            layer_list.append((np.argmax(np.array(temp_list[1:])) + 2 ))
        sess.close()
        tf.reset_default_graph()    
        return stats.mode(layer_list)[0][0]      
#-------------------------------------------
    
def dnn_fair_testing(dataset, sens_params, model_path):

    data = {"compas":compas_data}
           
    data_config = {"compas":compas}
    
    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    tf.set_random_seed(1234)
    layer_numbers=[]
    for trial in range(10):
        config = tf.ConfigProto(device_count = {'GPU': 0})
        config.allow_soft_placement= True

       # with tf.device('/CPU:1'):
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
        num_trial = 11
        num_rand_point_1 = 4 # number of random points in range(0 to mean)
        num_rand_point_2 = 3 # number of random points in range(mean to std)
        num_rand_point_3 = 1 # number of random points in range(mean + std to  mean + 2 * std)
    # Loading the result of QID
        layer_output = layer_out(sess,model,X.astype('float32'))
        input_df  = pd.read_csv('../results/' + dataset + '/OurTool/RQ3/total_disc_'+str(trial)+'.csv',header='infer')
        input_df = input_df.drop(columns=['Unnamed: 0'])
        sample_df = input_df.copy()
        sample_df_rand = sample_df.sample(n = 900,axis = 0,random_state = np.random.RandomState())
        sample_df_maxk = sample_df.sort_values(by = 'k',ascending=False).head(100)
        sample_df = pd.concat([sample_df_rand,sample_df_maxk])
        ini_k_samples = sample_df['k']
        sample_df = sample_df.drop(columns = ['sh_entropy', 'k', 'disc', 'min_entropy']) 
        samples   = sample_df.to_numpy()
        num_samples = len(samples)
        print(num_samples)
        layer_number = layer_locator(sess, model, model_path, sens_params, input_shape, nb_classes,
              dataset,data_config[dataset], samples)
        layer_numbers.append(layer_number)
        #-----------------------------
        update_df = layer_output['ReLU'+str((2*layer_number) -1 )]
        update_min  = np.min(update_df,axis=0)
        update_max  = np.max(update_df,axis=0)
        update_mean = np.mean(update_df,axis=0)
        update_std  = np.std(update_df,axis=0)
        update_list = []
        update_list.append(update_min)
        rand_point_1 = np.sort(np.random.random(num_rand_point_1))
        for i in range(len(rand_point_1)):
            update_list.append(rand_point_1[i] * update_mean)
        update_list.append(update_mean)
        rand_point_2 = np.sort(np.random.random(num_rand_point_2))
        for i in range(len(rand_point_2)):
            update_list.append(update_mean + rand_point_2[i] * update_std)
        update_list.append(update_mean + update_std +  np.random.random(num_rand_point_3)[0] * update_std)        
        update_list.append(update_max)
        layer_size   = model.layers[(layer_number*2) - 1].input_shape[1]
        layer_name   = model.layers[(layer_number*2) - 1]
        num_layers   = len(model.layers)

        all_dic = {}
        accu_neuron = {}
        acc_try = {}
        sample_ind = 0
        for sample in samples:       
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
            clear_output(wait=True)
            sample_ind += 1

        # create the folder for storing the fairness testing result
        if not os.path.exists('../results/'):
            os.makedirs('../results/')
        if not os.path.exists('../results/' + dataset + '/'):
            os.makedirs('../results/' + dataset + '/')
        if not os.path.exists('../results/' + dataset + '/OurTool/'):
            os.makedirs('../results/' + dataset + '/OurTool/')
        if not os.path.exists('../results/' + dataset + '/OurTool/RQ3/'):
            os.makedirs('../results/' + dataset + '/OurTool/RQ3/')          
        if not os.path.exists('../results/' + dataset + '/OurTool/RQ3/table2/'):
            os.makedirs('../results/' + dataset + '/OurTool/RQ3/table2/')
        np.save('../results/'+dataset+'/OurTool/RQ3/table2/inik_'+str(num_samples)+'_'+str(trial)+'.npy',
                np.array(ini_k_samples))
        np.save('../results/'+dataset+'/OurTool/RQ3/table2/accu_dic_'+str(num_samples)+'_'+str(trial)+'.npy',
                acc_try) 
        np.save('../results/'+dataset+'/OurTool/RQ3/table2/all_dic_'+str(num_samples)+'_'+str(trial)+'.npy', 
                all_dic)   

        accu_dic = dict(np.load('../results/'+dataset+'/OurTool/RQ3/table2/accu_dic_'+str(num_samples)+'_'+str(trial)+'.npy',
                                allow_pickle=True).item())  
        all_dic  = dict(np.load('../results/'+dataset+'/OurTool/RQ3/table2/all_dic_'+str(num_samples)+'_'+str(trial)+'.npy',
                               allow_pickle=True).item())
        ini_k = np.load('../results/'+dataset+'/OurTool/RQ3/table2/inik_'+str(num_samples)+'_'+str(trial)+'.npy')

        num_samples = len(all_dic.keys())
        num_force   = len(all_dic[0].keys())
        num_neuron  = len(all_dic[0][0].keys())
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
        acc_e   = 0.05
        for neuron in range(num_neuron):
            k_deact = df.loc[(df['neuron'] == neuron) & (df['force'] <= 1) & \
                                  (df['acc'] >= ini_acc - acc_e)]['K'].mean()
            k_act   = df.loc[(df['neuron'] == neuron) & (df['force'] > 1) & \
                                  (df['acc'] >= ini_acc - acc_e)]['K'].mean()
            k_init  = df.loc[(df['neuron'] == neuron) & (df['acc'] >= ini_acc - acc_e)]['init_k'].mean()
            R_act_temp   = (k_act - k_init) / k_init
            R_deact_temp = (k_deact - k_init) / k_init
            diff_R_temp  = R_act_temp - R_deact_temp
            R_act.append(R_act_temp)
            R_deact.append(R_deact_temp)
            diff_R.append(diff_R_temp)


        df.to_csv('../results/'+dataset+'/OurTool/RQ3/table2/df_'+str(num_samples)+'_'+str(trial)+'.csv',index=False)
        np.save('../results/'+dataset+'/OurTool/RQ3/table2/R_act_'+str(num_samples)+'_'+str(trial)+'.npy',R_act)
        np.save('../results/'+dataset+'/OurTool/RQ3/table2/R_deact_'+str(num_samples)+'_'+str(trial)+'.npy',R_deact)
        np.save('../results/'+dataset+'/OurTool/RQ3/table2/R_diffR_'+str(num_samples)+'_'+str(trial)+'.npy',diff_R)
    np.save('../results/'+dataset+'/OurTool/RQ3/table2/res_layers_'+str(num_samples)+'_'+str(trial)+'.npy',layer_numbers)
def main(argv = None):
    
    dnn_fair_testing(dataset = FLAGS.dataset, 
                     sens_params = FLAGS.sens_params,
                     model_path  = FLAGS.model_path)
    print(time.time() - time1 )

if __name__ == '__main__':
    flags.DEFINE_string("dataset", "compas", "the name of dataset")
    flags.DEFINE_string('model_path', '../models/', 'the path for testing model')
    flags.DEFINE_list('sens_params',[3,2,1],'sensitive parameters index.1 for age, 9 for gender, 8 for race')
    tf.app.run()

