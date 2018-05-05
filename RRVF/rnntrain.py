# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:32:28 2018

@author: Chen
"""

import pandas as pds
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import csv
import os
ops.reset_default_graph()
states1 = ['genre_name','month','day_of_week','area_name','holiday_flg']
states2 = ['store_id','genre_name','month','day_of_week','area_name','holiday_flg']
data = pds.read_csv('train_data.csv')
label = data['reserve_visitors'].values
label = np.expand_dims(label,axis = 1)
train_data = data.drop(['reserve_visitors','Unnamed: 0','day','longitude','latitude'],axis=1).reindex(columns=states1).values.astype(np.float32)
test_data = pds.read_csv('test_data.csv').drop(['Unnamed: 0','day','longitude','latitude'],axis=1).reindex(columns=states2)
test_id = test_data.values[:,:1]
test_data = test_data.values[:,1:].astype(np.int32)


hidden_size = 64
layer_num = 2
learning_rate = 0.01
#batch_size_dict = {0:512,1:1024,2:2048}
batch_size = 4096
with tf.name_scope('inputs'):
    x_data= tf.placeholder(shape = [1,None,5],dtype = tf.float32,name='x_input')
    y_target = tf.placeholder(shape = [None,1],dtype = tf.float32,name='y_input')
    keep_prob = tf.placeholder(tf.float32)

def unit_lstm():
    # 定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    #添加 dropout layer, 一般只设置 output_keep_prob
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return lstm_cell

mlstm_cell = tf.contrib.rnn.MultiRNNCell([unit_lstm() for i in range(layer_num)], state_is_tuple=True)

init_state = mlstm_cell.zero_state(1, dtype=tf.float32)

outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=x_data, initial_state=init_state, time_major=False)
h_state = outputs[:, -1, :]

with tf.name_scope('full'):
    with tf.name_scope('weight'):
        A4 = tf.Variable(tf.random_normal(shape=[hidden_size,1]))
        tf.summary.histogram('weight',A4)
    with tf.name_scope('bias'):
        b4 = tf.Variable(tf.random_normal(shape=[1]))
        tf.summary.histogram('bias',b4)
final_output = tf.add(tf.matmul(h_state,A4),b4)
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
avg_loss = 0
shuffled_ix = np.random.permutation(np.arange(len(train_data)))
train_data = train_data[shuffled_ix]
label = label[shuffled_ix] 
for i in range(10000):
    if (i+1)%1000 == 0:
        if learning_rate > 0.0001:
            learning_rate = learning_rate/10
    #batch_size = batch_size_dict[i//30000]
    rand_index = np.random.choice(len(train_data),replace=False,size = batch_size)
    rand_x = [train_data[rand_index]]
    rand_y = label[rand_index]
    summary = sess.run(merged,feed_dict = {x_data:rand_x,y_target:rand_y,keep_prob:0.5})
    train_writer.add_summary(summary, i+1)
    sess.run(train_step,feed_dict = {x_data:rand_x,y_target:rand_y,keep_prob: 0.5})
    temp_loss = sess.run(loss,feed_dict = {x_data:rand_x,y_target:rand_y,keep_prob: 0.5})
    avg_loss = (avg_loss*i+temp_loss)/(i+1)
    current_learning_rate = learning_rate
    #output = sess.run(final_output,feed_dict = {x_data:rand_x,y_target:rand_y})
    if (i+1)%10 == 0:
        print('Gen={}.LR ={:.3f}. Loss={:.3f}. Avg_loss={:.3f}.'.format(i+1,current_learning_rate,temp_loss,avg_loss))
        #print('Generation: {}. Loss = {:.3f}'.format(i+1,temp_loss))

filename = 'submission.csv'
for i in range(len(test_data)):
    data = np.expand_dims(test_id[i],axis = 0)+','+str(round(sess.run(final_output,feed_dict = {x_data:[np.expand_dims(test_data[i],axis = 0)],keep_prob:1.0})[0]))
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
    else:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data) 
f.close()