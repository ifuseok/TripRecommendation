"""
Author : Lee Won Seok
This GRU layer made for CoreML. CoreML can't use Pack ops,thus you can't contain tensorflow GRU_cell() in graph.
So I made this GRU code. This code only calculate forward process for GRU. Therefore it can replace tensorflow GRU_Cell().

Class_input -------------
input_dim : data input size (int)
hidden_size : hidden_size for Weight (int)
dropout : Use dropout or not?  (bool)
dropout_p_hidden : Determine dropout size (float)
-------------------------
output -----------------
h_t : output result of forward process
next_state : hidden state of forward process


For one layer GRU cell (But if you modify this code, you can use multi layer GRU.)
"""

import tensorflow as tf

class GRU:
    def __init__(self,input_dim,hidden_size,dropout,dropout_p_hiddden):
        self.dropout = dropout
        self.dropout_p_hidden = dropout_p_hiddden
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        #Weith for input
        with tf.variable_scope('input_'):
            self.Ur = tf.get_variable('reset_Weigth',[self.input_dim,self.hidden_size],initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
            self.Uz = tf.get_variable('Update_Weigth',[self.input_dim,self.hidden_size],initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
            self.Uh = tf.get_variable('hidden_Weigth', [self.input_dim, self.hidden_size],initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))

        #Weigth for state
        with tf.variable_scope('state_'):
            self.Wr = tf.get_variable('reset_Weigth',[self.hidden_size,self.hidden_size],initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
            self.Wz = tf.get_variable('update_Weigth', [self.hidden_size, self.hidden_size],initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            self.Wh = tf.get_variable('hidden_Weigth', [self.hidden_size, self.hidden_size],initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))

        with tf.variable_scope('bias_'):
            self.br = tf.get_variable('reset_bias',[self.hidden_size],initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
            self.bz = tf.get_variable('update_bias', [self.hidden_size],
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            self.bh = tf.get_variable('hidden_bias', [self.hidden_size],
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))


    def forward_process(self,X_input,State_input,hidden_act=tf.nn.tanh):
        z_t = tf.nn.sigmoid(tf.matmul(X_input,self.Uz) + tf.matmul(State_input,self.Wz)+self.bz) # batch_size,hidden_size
        r_t = tf.nn.sigmoid(tf.matmul(X_input,self.Ur) + tf.matmul(State_input,self.Wr)+self.br) # batch_size,hidden_size
        if self.dropout == True:
            z_t = tf.nn.dropout(z_t, keep_prob = self.dropout_p_hidden)
            r_t = tf.nn.dropout(r_t,keep_prob = self.dropout_p_hidden)
        h_t = hidden_act(tf.matmul(X_input, self.Uh) + tf.matmul(tf.multiply(State_input, r_t),self.Wh) + self.bh)  # batch_size,hidden_size
        if self.dropout == True:
            h_t = tf.nn.dropout(h_t,keep_prob=self.dropout_p_hidden)

        # computes state t
        next_state = tf.multiply((1-z_t),h_t) + tf.multiply(z_t,State_input)

        return h_t,next_state


