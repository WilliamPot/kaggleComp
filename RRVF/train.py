# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:57:49 2018

@author: Chen
"""

import pandas as pds
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import csv
import os
ops.reset_default_graph()
states1 = ['genre_name','month','day_of_week','longitude','latitude','holiday_flg']
states2 = ['store_id','genre_name','month','day_of_week','longitude','latitude','holiday_flg']
data = pds.read_csv('train_data.csv')
data = data.drop([455966,452073,157655])
label = data['reserve_visitors'].values
label = np.expand_dims(label,axis = 1)
train_data = data.drop(['reserve_visitors','Unnamed: 0','area_name','day'],axis=1).reindex(columns=states1).values.astype(np.float32)
test_data = pds.read_csv('test_data.csv').drop(['Unnamed: 0','area_name','day'],axis=1).reindex(columns=states2)
test_id = test_data.values[:,:1]
test_data = test_data.values[:,1:].astype(np.float32)


hidden_layer_nodes = 20
hidden_layer_nodes2 = 32
hidden_layer_nodes3 = 32
hidden_layer_nodes4 = 20

learning_rate = 0.1

ema = tf.train.ExponentialMovingAverage(decay=0.5)
def mean_var_with_update(ema,fc_mean,fc_var):
    ema_apply_op = ema.apply([fc_mean,fc_var])
    with tf.control_dependencies([ema_apply_op]):
        return tf.identity(fc_mean), tf.identity(fc_var)

#batch_size_dict = {0:512,1:1024,2:2048}
batch_size = 64
#with tf.name_scope('global_step'):
#    global_step = tf.Variable(0, trainable=False)
#    learning_rate = tf.train.exponential_decay(0.1, global_step, 2000, 0.96, staircase=True) 
with tf.name_scope('inputs'):
    x_data= tf.placeholder(shape = [None,6],dtype = tf.float32,name='x_input')
    y_target = tf.placeholder(shape = [None,1],dtype = tf.float32,name='y_input')
with tf.name_scope('layer1'):
    with tf.name_scope('weights1'):
        A1 = tf.Variable(tf.random_normal(shape=[6,hidden_layer_nodes]))
        tf.summary.histogram('weights1',A1)
    with tf.name_scope('bias1'):
        b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
        tf.summary.histogram('bias1',b1)
#hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data,A1),b1))

hidden_output = tf.add(tf.matmul(x_data,A1),b1)
fc_mean,fc_var = tf.nn.moments(hidden_output,axes=[0,1])
scale = tf.Variable(tf.ones([hidden_layer_nodes]))
shift = tf.Variable(tf.zeros([hidden_layer_nodes]))
epsilon = 0.001
mean, var = mean_var_with_update(ema,fc_mean,fc_var)
hidden_output = tf.nn.relu(tf.nn.batch_normalization(hidden_output,mean,var,shift,scale,epsilon))

with tf.name_scope('layer2'):
    with tf.name_scope('weight2'):
        A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,hidden_layer_nodes2]))
        tf.summary.histogram('weights2',A2)
    with tf.name_scope('bias2'):
        b2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes2]))
        tf.summary.histogram('bias2',b2)
#hidden_output2 = tf.nn.relu(tf.add(tf.matmul(hidden_output,A2),b2))
     
hidden_output2 = tf.add(tf.matmul(hidden_output,A2),b2)
fc_mean,fc_var = tf.nn.moments(hidden_output2,axes=[0,1])
scale = tf.Variable(tf.ones([hidden_layer_nodes2]))
shift = tf.Variable(tf.zeros([hidden_layer_nodes2]))
epsilon = 0.001
mean, var = mean_var_with_update(ema,fc_mean,fc_var)
hidden_output2 = tf.nn.relu(tf.nn.batch_normalization(hidden_output2,mean,var,shift,scale,epsilon))

with tf.name_scope('layer3'):
    with tf.name_scope('weight3'):
        A3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes2,hidden_layer_nodes3]))
        tf.summary.histogram('weights3',A3)
    with tf.name_scope('bias3'):
        b3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes3]))
        tf.summary.histogram('bias3',b3)
#hidden_output3 = tf.nn.relu(tf.add(tf.matmul(hidden_output2,A3),b3))  

hidden_output3 = tf.add(tf.matmul(hidden_output2,A3),b3)
fc_mean,fc_var = tf.nn.moments(hidden_output3,axes=[0,1])
scale = tf.Variable(tf.ones([hidden_layer_nodes3]))
shift = tf.Variable(tf.zeros([hidden_layer_nodes3]))
epsilon = 0.001
mean, var = mean_var_with_update(ema,fc_mean,fc_var)
hidden_output3 = tf.nn.relu(tf.nn.batch_normalization(hidden_output3,mean,var,shift,scale,epsilon))

with tf.name_scope('layer4'):
    with tf.name_scope('weight4'):
        A4 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes3,hidden_layer_nodes4]))
        tf.summary.histogram('weights4',A4)
    with tf.name_scope('bias4'):
        b4 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes4]))
        tf.summary.histogram('bias4',b4)
#hidden_output4 = tf.nn.relu(tf.add(tf.matmul(hidden_output3,A4),b4)) 
  

hidden_output4 = tf.add(tf.matmul(hidden_output3,A4),b4)
fc_mean,fc_var = tf.nn.moments(hidden_output4,axes=[0,1])
scale = tf.Variable(tf.ones([hidden_layer_nodes4]))
shift = tf.Variable(tf.zeros([hidden_layer_nodes4]))
epsilon = 0.001
mean, var = mean_var_with_update(ema,fc_mean,fc_var)
hidden_output4 = tf.nn.relu(tf.nn.batch_normalization(hidden_output4,mean,var,shift,scale,epsilon))

with tf.name_scope('layer5'):
    with tf.name_scope('weight5'):
        A5 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes4,1]))
        tf.summary.histogram('weights5',A5)
    with tf.name_scope('bias5'):
        b5 = tf.Variable(tf.random_normal(shape=[1]))
        tf.summary.histogram('bias5',b5)
final_output = tf.add(tf.matmul(hidden_output4,A5),b5)
#final_output = tf.nn.dropout(final_output, keep_prob) 
loss= tf.reduce_mean(tf.square(y_target-final_output))
tf.summary.scalar('loss',loss)
my_opt= tf.train.AdamOptimizer(learning_rate)
#train_step = my_opt.minimize(loss,global_step=global_step)
train_step = my_opt.minimize(loss)
init=tf.global_variables_initializer()
#config = tf.ConfigProto() 
#config.gpu_options.per_process_gpu_memory_fraction = 0.8 # 占用GPU40%的显存 
#sess = tf.Session(config=config)
sess = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('tensorboard',sess.graph)
sess.run(init)
shuffled_ix = np.random.permutation(np.arange(len(train_data)))
train_data = train_data[shuffled_ix]
label = label[shuffled_ix]
loss_queue = []
for i in range(500000):
    if (i+1)%60000 == 0:
        if learning_rate > 0.00001:
            learning_rate = learning_rate/10
    #batch_size = batch_size_dict[i//30000]
    rand_index = np.random.choice(len(train_data),replace=False,size = batch_size)
    rand_x = train_data[rand_index]
    rand_y = label[rand_index]
    summary = sess.run(merged,feed_dict = {x_data:rand_x,y_target:rand_y})
    train_writer.add_summary(summary, i+1)
    sess.run(train_step,feed_dict = {x_data:rand_x,y_target:rand_y})
    temp_loss = sess.run(loss,feed_dict = {x_data:rand_x,y_target:rand_y})
    current_learning_rate = learning_rate
    if len(loss_queue) < 5000:
        loss_queue.append(temp_loss)
    else:
        loss_queue.pop(0)
        loss_queue.append(temp_loss)
    #output = sess.run(final_output,feed_dict = {x_data:rand_x,y_target:rand_y})
    if (i+1)%100 == 0:
        print('Gen={}.LR ={:.3f}. Loss={:.3f}. Avg_loss={:.3f}.'.format(i+1,current_learning_rate,temp_loss,np.mean(loss_queue)))
        #print('Generation: {}. Loss = {:.3f}'.format(i+1,temp_loss))

filename = 'submission.csv'
for i in range(len(test_data)):
    data = np.expand_dims(test_id[i],axis = 0)+','+str(round(sess.run(final_output,feed_dict = {x_data:np.expand_dims(test_data[i],axis = 0)})[0][0]))
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
    else:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data) 
f.close()
        