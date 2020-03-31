"""
Author : Lee Won Seok
CoreML doesn't supply 'Pack' operation. So it is hard convert tensor to CoreML.
This code only use several 'ops' that CoreML supply.
Coreml can't use tf.concat function.n
Output is embedding weight.
"""

import tensorflow as tf
from .gru import GRU


def Resave_model(item_table,attribute,*, ID, tag, city, states, ckpt_dir, model_number, rnn_size, item_size):
    # This graph restoring weight from model and save weight in valuable#
    g = tf.Graph()
    sess1 = tf.Session(graph=g)
    # Graph build
    with g.as_default():
        with tf.variable_scope('gru_layer'):
            softmax_Ws = []
            softmax_bs = []
            embeddings = []

            for i in range(len(attribute)):
                softmax_Ws.append(tf.get_variable('softmax_w'+str(i),[item_size[i],rnn_size[i]],initializer=tf.random_normal_initializer))
                softmax_bs.append(tf.get_variable('softmax_b'+str(i),[item_size[i],1],initializer=tf.zeros_initializer()))
                embeddings.append(tf.get_variable('embedding'+str(i),[item_size[i],rnn_size[i]],initializer=tf.random_normal_initializer()))

            softmax_ew = tf.nn.softmax(tf.get_variable('softmax_ew', [len(rnn_size)],
                                                       initializer=tf.ones_initializer()))
            softmax_bw = tf.nn.softmax(tf.get_variable('softmax_bw', [len(rnn_size)],
                                                       initializer=tf.ones_initializer()))
            embedding_lookup = []
            softmax_w_lookup = []
            softmax_b_lookup = []

            # embedding look up from each weight layer
            for i in range(len(attribute)):
                embedding_lookup.append(tf.nn.embedding_lookup(embeddings[i],item_table[attribute[i]]))
                softmax_w_lookup.append(tf.nn.embedding_lookup(softmax_Ws[i],item_table[attribute[i]])* softmax_ew[i])
                softmax_b_lookup.append(tf.nn.embedding_lookup(softmax_bs[i],item_table[attribute[i]]))
            # concat model concat embedding weights
            total_embed = tf.concat(embedding_lookup,axis=1)
            total_softw = tf.concat(softmax_w_lookup,axis=1)
            total_softb = tf.concat(softmax_b_lookup,axis=1)

            b_final = tf.matmul(total_softb, tf.reshape(softmax_bw, (len(attribute), 1)))


        sess1.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess1, "{}/attribute_model-{}".format(ckpt_dir, model_number))
    # weight restore and assign in new variable
    embed,soft_w, soft_b = sess1.run([total_embed,total_softw,b_final])

    # build mini model (Only use several ops)
    tf.reset_default_graph()
    sess = tf.Session()
    X = tf.placeholder(tf.float32, [1, sum(rnn_size)], name="input")
    state = tf.placeholder(tf.float32, [1, sum(rnn_size)], name='rnn_state')
    with tf.variable_scope('gru_layer'):
        softmax_W = tf.convert_to_tensor(soft_w.T)
        softmax_b = tf.convert_to_tensor(soft_b.T)

        ## RNN foward process
        cell = GRU(input_dim=sum(rnn_size), hidden_size=sum(rnn_size), dropout=False, dropout_p_hiddden=1.0)

        outputs, state_history = cell.forward_process(X, state)

    ## softmax layer, calculate next item probability

    result = tf.add(tf.matmul(outputs, softmax_W), softmax_b,name="result2")

    ##  This code exists to search what you want from the model results. (ex city address or category)
    ID_tensor = tf.multiply(tf.convert_to_tensor(ID), tf.cast(result/result,tf.int32), name="ID")
    tag_tensor = tf.multiply(tf.convert_to_tensor(tag), tf.cast(result/result,tf.int32), name="category")
    city_tensor = tf.multiply(tf.convert_to_tensor(city), tf.cast(result/result,tf.int32), name="city")
    state_tensor = tf.multiply(tf.convert_to_tensor(states), tf.cast(result/result,tf.int32), name="states")

    #result = tf.reshape(result, (item_size, 1))
    #cc=tf.identity(outputs, name="out")
    #result2 = tf.concat([result, ID_tensor, tag_tensor, city_tensor, state_tensor], axis=1, name="result2")

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, "{}/attribute_model-{}".format(ckpt_dir, model_number))

    saver.save(sess, "{}/custom_model".format(ckpt_dir))
    print("{}/custom_model 로 모델 저장 완료 및 embedding input 복원!".format(ckpt_dir))
    return embed # this embed is input of coreml model
