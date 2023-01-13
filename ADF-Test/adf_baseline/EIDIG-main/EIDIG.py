"""
This python file implement our approach EIDIG, and it can be simply applied to other differentiable prediction models.
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import cluster
import itertools
import time
import generation_utilities
from adf_utils.utils_tf import  model_argmax,model_prediction
global start_time
from Timeout import timeout
global global_disc, total_global
global total_local, local_disc
total_global=set()
global_disc=set()
total_local=set()
local_disc=set()
def compute_grad(sess,preds,z,grad_0,x, model):
    # compute the gradient of model perdictions w.r.t input attributes
    gradient = sess.run(grad_0, feed_dict={z: np.array([x])})
#     x = tf.constant([x], dtype=tf.float32)
#     with tf.GradientTape() as tape:
#         tape.watch(x)
#         y_pred = model(x)
#     gradient = tape.gradient(y_pred, x)
    if model_prediction(sess, z, preds, np.array([x]))[0][1]>0.5:
        return gradient[0]
        
    else:
        return (-gradient[0])
        
    #return gradient[0].numpy() if model(x) > 0.5 else -gradient[0].numpy()





def global_generation(start_time,sess,preds,z,grad_0,X, seeds, num_attribs, protected_attribs, constraint, model, decay, max_iter, s_g):
    # global generation phase of EIDIG
    global total_global,global_disc
    
    time_to_first=-1
    g_id = np.empty(shape=(0, num_attribs))
    all_gen_g = np.empty(shape=(0, num_attribs))
    try_times = 0
    g_num = len(seeds)
    for i in range(g_num):
        
        x1 = seeds[i].copy()
        grad1 = np.zeros_like(X[0]).astype(float)
        grad2 = np.zeros_like(X[0]).astype(float)
        for _ in range(max_iter):
            if time.time()-start_time>=900:
                print(time.time()-start_time)
                break
            try_times += 1
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            temp=x1.copy()
            temp=temp.astype('int')
            temp[0]=0
            if tuple(temp) not in total_global and tuple(temp) not in total_local:
                    total_global.add(tuple(temp))
            if generation_utilities.is_discriminatory(sess,preds,z,x1, similar_x1, model):
                temp=x1.copy()
                temp=temp.astype('int')
                temp[0]=0

                if (tuple(temp) not in global_disc) and (tuple(temp) not in local_disc):
                    if len(global_disc)==1:
                        time_to_first=time.time()-start_time
                    global_disc.add(tuple(temp))
                    
                g_id = np.append(g_id, [x1], axis=0)
                
                break
            x2 = generation_utilities.max_diff(sess,preds,z,x1, similar_x1, model)
            grad1 = decay * grad1 + compute_grad(sess,preds,z,grad_0,x1, model)
            grad2 = decay * grad2 + compute_grad(sess,preds,z,grad_0,x2, model)
            direction = np.zeros_like(X[0])
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attrib in range(num_attribs):
                if attrib not in protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                    direction[attrib] = (-1) * sign_grad1[attrib]
            x1 = x1 + s_g * direction
            x1 = generation_utilities.clip(x1, constraint)
            all_gen_g = np.append(all_gen_g, [x1], axis=0)
    print('len',len(g_id),len(global_disc))
    g_id = np.array(list(set([tuple(id) for id in g_id])))
    return g_id, all_gen_g, try_times, len(total_global), len(global_disc),time_to_first


def local_generation(start_time,sess,preds,z,grad_0,num_attribs, l_num, g_id, protected_attribs, constraint, model, update_interval, s_l, epsilon):
    # local generation phase of EIDIG
    global total_local,local_disc
    direction = [-1, 1]
    l_id = np.empty(shape=(0, num_attribs))
    all_gen_l = np.empty(shape=(0, num_attribs))
    try_times = 0
    for x1 in g_id:
        if time.time()-start_time>=900:
            print(time.time()-start_time)
            break
        x0 = x1.copy()
        similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
        x2 = generation_utilities.max_diff(sess,preds,z,x1, similar_x1, model)
        grad1 = compute_grad(sess,preds,z,grad_0,x1, model)
        grad2 = compute_grad(sess,preds,z,grad_0,x2, model)
        p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
        p0 = p.copy()
        suc_iter = 0
        for _ in range(l_num):
            if time.time()-start_time>=900:
                print(time.time()-start_time)
                break
            try_times += 1
            if suc_iter >= update_interval:
                similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
                x2 = generation_utilities.find_pair(x1, similar_x1, model)
                grad1 = compute_grad(sess,preds,z,grad_0,x1, model)
                grad2 = compute_grad(sess,preds,z,grad_0,x2, model)
                p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
                suc_iter = 0
            suc_iter += 1
            a = generation_utilities.random_pick(p)
            s = generation_utilities.random_pick([0.5, 0.5])
            x1[a] = x1[a] + direction[s] * s_l
            x1 = generation_utilities.clip(x1, constraint)
            all_gen_l = np.append(all_gen_l, [x1], axis=0)
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            temp=x1.copy()
            temp=temp.astype('int')
            temp[0]=0
            if (tuple(temp) not in total_global) and (tuple(temp) not in total_local):
                    
                    total_local.add(tuple(temp))
            if generation_utilities.is_discriminatory(sess,preds,z,x1, similar_x1, model):
                temp=x1.copy()
                temp=temp.astype('int')
                temp[0]=0

                if (tuple(temp) not in global_disc) and (tuple(temp) not in local_disc):
                    
                    local_disc.add(tuple(temp))
                l_id = np.append(l_id, [x1], axis=0)
            else:
                x1 = x0.copy()
                p = p0.copy()
                suc_iter = 0
    print('len loc',len(l_id),len(local_disc))
    l_id = np.array(list(set([tuple(id) for id in l_id])))
    return l_id, all_gen_l, try_times, len(total_local),len(local_disc)
    
#@timeout(int(900))
def individual_discrimination_generation(start_time,sess,preds,z,grad_0,X, seeds, protected_attribs, constraint, model, decay, l_num, update_interval, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6):
    # complete implementation of EIDIG
    # return non-duplicated individual discriminatory instances generated, non-duplicate instances generated and total number of search iterations

    num_attribs = len(X[0])
    g_id, gen_g, g_gen_num, tot_g, g_disc,t_f_g = global_generation(start_time,sess,preds,z,grad_0,X, seeds, num_attribs, protected_attribs, constraint, model, decay, max_iter, s_g)
    l_id, gen_l, l_gen_num, tot_l, l_disc = local_generation(start_time,sess,preds,z,grad_0,num_attribs, l_num, g_id, protected_attribs, constraint, model, update_interval, s_l, epsilon_l)
    all_id = np.append(g_id, l_id, axis=0)
    all_gen = np.append(gen_g, gen_l, axis=0)
    all_id_nondup = np.array(list(set([tuple(id) for id in all_id])))
    all_gen_nondup = np.array(list(set([tuple(gen) for gen in all_gen])))
    return all_id_nondup, all_gen_nondup, g_gen_num + l_gen_num,tot_g,g_disc,tot_l,l_disc,t_f_g


def seedwise_generation(X, seeds, protected_attribs, constraint, model, l_num, decay, update_interval, max_iter=10, s_g=1.0, s_l=1.0, epsilon=1e-6):
    # perform global generation and local generation successively on each single seed

    num_seeds = len(seeds)
    num_gen = np.array([0] * num_seeds)
    num_ids = np.array([0] * num_seeds)
    num_attribs = len(X[0])
    ids = np.empty(shape=(0, num_attribs))
    all_gen = np.empty(shape=(0, num_attribs))
    direction = [-1, 1]
    for index, instance in enumerate(seeds):
        x1 = instance.copy()
        flag = False
        grad1 = np.zeros_like(X[0]).astype(float)
        grad2 = np.zeros_like(X[0]).astype(float)
        for j in range(max_iter):
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                ids = np.append(ids, [x1], axis=0)
                flag = True
                break
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            grad1 = decay * grad1 + compute_grad(x1, model)
            grad2 = decay * grad2 + compute_grad(x2, model)
            direction_g = np.zeros_like(X[0])
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attrib in range(num_attribs):
                if attrib not in protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                    direction_g[attrib] = (-1) * sign_grad1[attrib]
            x1 = x1 + s_g * direction_g
            x1 = generation_utilities.clip(x1, constraint)
            all_gen = np.append(all_gen, [x1], axis=0)
        if flag == True:
            x0 = x1.copy()
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            grad1 = compute_grad(x1, model)
            grad2 = compute_grad(x2, model)
            p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
            p0 = p.copy()
            suc_iter = 0
            for _ in range(l_num):
                if suc_iter >= update_interval:
                    similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
                    x2 = generation_utilities.find_pair(x1, similar_x1, model)
                    grad1 = compute_grad(x1, model)
                    grad2 = compute_grad(x2, model)
                    p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
                    suc_iter = 0
                suc_iter += 1
                a = generation_utilities.random_pick(p)
                s = generation_utilities.random_pick([0.5, 0.5])
                x1[a] = x1[a] + direction[s] * s_l
                x1 = generation_utilities.clip(x1, constraint)
                all_gen = np.append(all_gen, [x1], axis=0)
                similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
                if generation_utilities.is_discriminatory(x1, similar_x1, model):
                    ids = np.append(ids, [x1], axis=0)
                else:
                    x1 = x0.copy()
                    p = p0.copy()
                    suc_iter = 0
        nondup_ids = np.array(list(set([tuple(id) for id in ids])))
        nondup_gen = np.array(list(set([tuple(gen) for gen in all_gen])))
        num_gen[index] = len(nondup_gen)
        num_ids[index] = len(nondup_ids)
    return num_gen, num_ids


def time_record(X, seeds, protected_attribs, constraint, model, decay, l_num, record_step, record_frequency, update_interval, max_iter=10, s_g=1.0, s_l=1.0, epsilon=1e-6):
    # record time consumption
    
    num_attribs = len(X[0])
    t = np.array([0.0] * record_frequency)
    direction_l = [-1, 1]
    threshold = record_step
    index = 0
    t1 = time.time()
    ids = np.empty(shape=(0, num_attribs))
    num_ids = num_ids_before = 0
    for instance in seeds:
        if num_ids >= record_frequency * record_step:
            break
        x1 = instance.copy()
        flag = False
        grad1 = np.zeros_like(X[0]).astype(float)
        grad2 = np.zeros_like(X[0]).astype(float)
        for i in range(max_iter+1):
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                ids = np.append(ids, [x1], axis=0)
                flag = True
                break
            if i == max_iter:
                break
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            grad1 = decay * grad1 + compute_grad(x1, model)
            grad2 = decay * grad2 + compute_grad(x2, model)
            direction_g = np.zeros_like(X[0])
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attrib in range(num_attribs):
                if attrib not in protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                    direction_g[attrib] = (-1) * sign_grad1[attrib]
            x1 = x1 + s_g * direction_g
            x1 = generation_utilities.clip(x1, constraint)
            t2 = time.time()
        if flag == True:
            ids = np.array(list(set([tuple(id) for id in ids])))
            num_ids = len(ids)
            if num_ids > num_ids_before:
                num_ids_before = num_ids
                if num_ids == threshold:
                    t[index] = t2 - t1
                    threshold += record_step
                    index += 1
                    if num_ids >= record_frequency * record_step:
                        break
            x0 = x1.copy()
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            grad1 = compute_grad(x1, model)
            grad2 = compute_grad(x2, model)
            p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
            p0 = p.copy()
            suc_iter = 0
            for _ in range(l_num):
                if suc_iter >= update_interval:
                    similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
                    x2 = generation_utilities.find_pair(x1, similar_x1, model)
                    grad1 = compute_grad(x1, model)
                    grad2 = compute_grad(x2, model)
                    p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
                    suc_iter = 0
                suc_iter += 1
                a = generation_utilities.random_pick(p)
                s = generation_utilities.random_pick([0.5, 0.5])
                x1[a] = x1[a] + direction_l[s] * s_l
                x1 = generation_utilities.clip(x1, constraint)
                t2 = time.time()
                similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
                if generation_utilities.is_discriminatory(x1, similar_x1, model):
                    ids = np.append(ids, [x1], axis=0)
                    ids = np.array(list(set([tuple(id) for id in ids])))
                    num_ids = len(ids)
                    if num_ids > num_ids_before:
                        num_ids_before = num_ids
                        if num_ids == threshold:
                            t[index] = t2 - t1
                            threshold += record_step
                            index += 1
                            if num_ids >= record_frequency * record_step:
                                break
                else:
                    x1 = x0.copy()
                    p = p0.copy()
                    suc_iter = 0
    return t