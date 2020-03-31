"""
Author : Lee Won Seok (reference 'GRU4REC_Tensorflow' ,Author : Weiping Song)
This code is parellel GRU model.This model learn weight of each attribute model.
"""


import os
import tensorflow as tf
import pandas as pd
import numpy as np
from random import shuffle
from .gru import GRU
from sklearn.preprocessing import minmax_scale

class GRU4Rec:
    def __init__(self, sess, args):
        self.sess = sess
        self.is_trainging = args.is_training
        self.layers = args.layers # must list  [1,1]
        self.rnn_size = args.rnn_size # must list example [[11,32,4,],[500]]
        self.n_epochs = args.n_epochs #
        self.batch_size = args.batch_size
        self.dropout_p_hidden = args.dropout_p_hidden
        self.learning_rate = args.learning_rate
        self.decay = args.decay
        self.decay_steps = args.decay_steps
        self.reset_after_session = args.reset_after_session
        self.session_key = args.session_key
        self.attribute_key = args.attribute_key
        self.sigma = args.sigma
        self.model_name = args.model_name
        self.time_key = args.time_key
        self.init_as_normal = args.init_as_normal
        self.item_key = args.item_key
        self.grad_cap = args.grad_cap
        self.attribute = args.attribute
        self.max_to_keep = args.max_ckpt_keep
        self.predict_states = [] # 여러모델 받아들일 거니까 필요함
        for i in range(len(self.model_name)):
            self.predict_states.append(np.zeros([self.batch_size,self.rnn_size[i]],dtype=np.float32))


        self.item_table = args.item_table
        self.test_model_train = args.test_model_train # 체크포인트 번호 넣어야함
        self.graph = args.graph # self.graph 내가 넣어야함 새로 학습할 모델 그래프

        if args.hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif args.hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        if args.loss == 'cross-entropy':
            if args.final_act == 'tanh':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy
        elif args.loss == 'bpr':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activation = self.relu
            elif args.final_act == 'tanh':
                self.final_activation = self.tanh
            elif args.final_act == 'scale':
                self.final_activation = self.scale
            else:
                raise NotImplementedError
            self.loss_function = self.bpr
        elif args.loss == 'top1':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activation = self.relu
            else:
                raise NotImplementedError
            self.loss_function = self.top1
        else:
            raise NotImplementedError

        self.states = []
        self.Xs = []
        self.Ys = []

        self.checkpoint_dir_train = args.checkpoint_dir_train ## attribute model, item model
        self.checkpoint_dir = args.checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            print('Checkpoint Dir not found I make it')
            os.makedirs(self.checkpoint_dir)
       #미리학습된 모델의 영역#
        graph = []
        self.sesses = []
        self.final_states = []
        self.yhats = []

        for i in range(len(self.model_name)):
            graph.append(tf.Graph())
            if i==0:
                with graph[i].as_default():

                    self.build_model(i)
                    sess = tf.Session(graph=graph[i])
                    #sess.run(tf.global_variables_initializer())
                    saver = tf.train.Saver(tf.global_variables(scope='gru_layer'))
                    ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir_train[i])
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess,'{}/attribute_model_{}-{}'.format(self.checkpoint_dir_train[i],self.attribute_key[i],self.test_model_train[i]))
                    self.sesses.append(sess)
                    print("완료")
            else:
                print("이거는?")
                with graph[i].as_default():
                    self.build_model2(i)
                    sess = tf.Session(graph=graph[i])
                    #sess.run(tf.global_variables_initializer())
                    saver = tf.train.Saver(tf.global_variables(scope='gru_layer'))
                    ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir_train[i])
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess,'{}/attribute_model-{}'.format(self.checkpoint_dir_train[i],self.test_model_train[i]))
                    self.sesses.append(sess)
                    print("완료")

        #######새로학습할 모델의 영역######
        with self.graph.as_default():
            self.weight_model()
            if self.is_trainging:
                self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(tf.global_variables(scope='main'), max_to_keep=self.max_to_keep)
            if self.is_trainging:
                return  # 훈련중이면 여기서 멈추게

            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, '{}/weight-model--{}'.format(self.checkpoint_dir,args.test_model))

        ##########################ACTIVATE FUNCTION##################################
    def linear(self, X):
        return X
    def tanh(self, X):
        return tf.nn.tanh(X)
    def scale(self,X):
        return minmax_scale(X,axis=1)
    def softmax(self, X):
        return tf.nn.softmax(X)
    def softmaxth(self, X):
        return tf.nn.softmax(tf.tanh(X))
    def relu(self, x):
        return tf.nn.relu(x)
    def sigmoid(self, x):
        return tf.nn.sigmoid(x)
    #############################LOSS FUNCTION#############################################33
    def cross_entropy(self, yhat):
        return tf.reduce_mean(-tf.log(tf.diag_part(yhat) + 1e-24))

    def bpr(self, yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat) - yhatT)))

    def top1(self, yhat):
        yhatT = tf.transpose(yhat)
        term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat) + yhatT) + tf.nn.sigmoid(yhatT ** 2), axis=0)
        term2 = tf.nn.sigmoid(tf.diag_part(yhat) ** 2) / self.batch_size
        return tf.reduce_mean(term1 - term2)

    def build_model(self,idx):

        X = tf.placeholder(tf.int32, [self.batch_size], name='input')
        Y = tf.placeholder(tf.int32, [self.batch_size], name='output')
        state = tf.placeholder(tf.float32, [self.batch_size, self.rnn_size[idx]], name='rnn_state')

        #self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.Xs.append(X)
        self.Ys.append(Y)
        self.states.append(state)

        with tf.variable_scope('gru_layer'):

            embedding = tf.get_variable('embedding_' + self.attribute_key[idx], [self.attribute[idx], self.rnn_size[idx]], initializer=tf.random_normal_initializer())
            softmax_W = tf.get_variable('softmax_w_' + self.attribute_key[idx], [self.attribute[idx], self.rnn_size[idx]], initializer=tf.random_normal_initializer())
            softmax_b = tf.get_variable('softmax_b_' + self.attribute_key[idx], [self.attribute[idx]], initializer=tf.constant_initializer(0.0))

            cell = GRU(input_dim=self.rnn_size[idx], hidden_size=self.rnn_size[idx], dropout=True,
                       dropout_p_hiddden=self.dropout_p_hidden)


            inputs = tf.nn.embedding_lookup(embedding, X)
            outputs, state = cell.forward_process(inputs, state)
            self.final_states.append(state)

        if self.is_trainging:
            '''
            Use other examples of the minibatch as negative samples.
            '''
            sampled_W = tf.nn.embedding_lookup(softmax_W, Y)  # y 배치 뽑기
            sampled_b = tf.nn.embedding_lookup(softmax_b, Y)
            logits = tf.matmul(outputs, sampled_W, transpose_b=True) + sampled_b  # batch_size x batch_size
            self.yhats.append(self.final_activation(logits))
        else:
            softmax_W_total = tf.nn.embedding_lookup(softmax_W,self.item_table[self.attribute_key[idx]])
            softmax_b_total = tf.nn.embedding_lookup(softmax_b,self.item_table[self.attribute_key[idx]])

            logits = tf.matmul(outputs, softmax_W_total,
                               transpose_b=True) + softmax_b_total  # (batch_sizexrnn) x (rnn_Size x n_item\1\5)
            self.yhats.append(self.final_activation(logits))

    def build_model2(self,idx):

        X = tf.placeholder(tf.int32, [self.batch_size], name='input')
        Y = tf.placeholder(tf.int32, [self.batch_size], name='output')
        state = tf.placeholder(tf.float32, [self.batch_size, self.rnn_size[idx]], name='rnn_state')

        #self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.Xs.append(X)
        self.Ys.append(Y)
        self.states.append(state)

        with tf.variable_scope('gru_layer'):

            embedding = tf.get_variable('embedding0', [self.attribute[idx], self.rnn_size[idx]], initializer=tf.random_normal_initializer())
            softmax_W = tf.get_variable('softmax_w0', [self.attribute[idx], self.rnn_size[idx]], initializer=tf.random_normal_initializer())
            softmax_b = tf.get_variable('softmax_b0', [self.attribute[idx],1], initializer=tf.constant_initializer(0.0))

            cell = GRU(input_dim=self.rnn_size[idx], hidden_size=self.rnn_size[idx], dropout=True,
                       dropout_p_hiddden=self.dropout_p_hidden)


            inputs = tf.nn.embedding_lookup(embedding, X)
            outputs, state = cell.forward_process(inputs, state)

            state = tf.contrib.layers.batch_norm(state,
                                                 center=True, scale=True,
                                                 is_training=False)
            self.final_states.append(state)

        if self.is_trainging:
            '''
            Use other examples of the minibatch as negative samples.
            '''
            sampled_W = tf.nn.embedding_lookup(softmax_W, Y)  # y 배치 뽑기
            sampled_b = tf.nn.embedding_lookup(softmax_b, Y)
            logits = tf.matmul(outputs, sampled_W, transpose_b=True) + sampled_b  # batch_size x batch_size
            self.yhats.append(self.final_activation(logits))
        else:
            softmax_W_total = tf.nn.embedding_lookup(softmax_W,self.item_table[self.attribute_key[idx]])
            softmax_b_total = tf.nn.embedding_lookup(softmax_b,self.item_table[self.attribute_key[idx]])

            logits = tf.matmul(outputs, softmax_W_total,
                               transpose_b=True) + tf.transpose(softmax_b_total)  # (batch_sizexrnn) x (rnn_Size x n_item\1\5)
            self.yhats.append(self.final_activation(logits))




    def weight_model(self):
        self.global_step = tf.Variable(0, name='global_step',trainable=False)
        self.rating = tf.placeholder(tf.float32,[len(self.model_name),None,None],name='input_yhat')
        with tf.variable_scope('main'):
            self.weight = tf.nn.softmax(tf.get_variable('yhat_w',[len(self.model_name)],initializer=tf.ones_initializer()))
            self.yhat2 = self.rating[0] * tf.nn.embedding_lookup(self.weight,0)

            for idx in range(1,len(self.model_name)):
                weight = tf.nn.embedding_lookup(self.weight,idx)
                self.yhat2 += self.rating[idx] * weight

            if self.is_trainging:
                self.cost = self.loss_function(self.yhat2)

        if not self.is_trainging: # 훈련용이 아니면 여기서 스텝 종료 #
            return
        self.lr = tf.maximum(1e-5, tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                              self.decay, staircase=True))

        """
        Optimizer
        """
        optimizer = tf.train.AdamOptimizer(self.lr)
        tvars = tf.trainable_variables()
        gvs = optimizer.compute_gradients(self.cost, tvars)

        if self.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in gvs]
        else:
            capped_gvs = gvs
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)


    def init(self,data):
        uni=data[self.session_key].unique() # 추가
        shuffle(uni) #추가
        new = pd.DataFrame(uni, columns=[self.session_key]) #추가
        new['sort'] = range(0, uni.shape[0]) #추가
        data = new.merge(data, how='inner', on=self.session_key) # 추가
        data.sort_values(['sort',self.time_key],inplace=True)
        offset_sessions = np.zeros(data[self.session_key].nunique()+1,dtype=np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum() # session key기준으로 누적합계 구하기
        return offset_sessions,data

    def fit(self, data2):
        '''
        item_table frame in columns
        item_id attribute1 attribute2 ....
        '''
        self.error_during_train = False  # ?? 어디에 쓰이노
        print('fitting model....')
        for epoch in list(range(self.n_epochs)):
            offset_sessions,data = self.init(data2)
            print(offset_sessions)
            epoch_cost = []
            states = []
            for i in range(len(self.attribute_key)):
                states.append(np.zeros([self.batch_size,self.rnn_size[i]],dtype=np.float32))

            session_idx_arr = np.arange(len(offset_sessions) - 1)  # session index 담긴 어레이
            iters = np.arange(self.batch_size)  # 0~batchsize-1 까지
            maxiter = iters.max()
            start = offset_sessions[session_idx_arr[iters]]  # 0번 부터 49번 까지 start 에 담고 데이터안에서 각 세션 시작 인덱스를 담고있음
            end = offset_sessions[session_idx_arr[iters] + 1]  # 1번 부터 50번 까지 end에 담음 앞에 각 세션이 끝나는 시점의 인덱스
            finished = False
            while not finished:
                minlen = (end - start).min()  # 전체 세션중 최소길이#
                out_idx = data[self.attribute_key].values[start]

                for i in range(minlen - 1):  # 첫 값은 out으로 이용되지 않음
                    in_idx = out_idx
                    out_idx = data[self.attribute_key].values[start + i]
                    fetches = []
                    feed_dicts = []
                    for i in range(len(self.attribute_key)):
                        fetches.append([self.yhats[i],self.final_states[i]])
                        feed_dict = {self.Xs[i]: in_idx[:,i], self.Ys[i]: out_idx[:,i]}
                        feed_dict[self.states[i]] = states[i]
                        feed_dicts.append(feed_dict)
                    yhats = []
                    at_states = []
                    for idx,ses in enumerate(self.sesses):
                        y,at_state= ses.run(fetches[idx], feed_dict=feed_dicts[idx])
                        yhats.append((self.scale(y)+1))
                        #yhats.append(y)
                        at_states.append(at_state)
                    yhats = np.array(yhats)
                    fetche_rate = [self.cost,self.global_step,self.lr,self.train_op,self.weight]
                    cost, step, lr, _ ,weight= self.sess.run(fetche_rate, feed_dict={self.rating : yhats})
                    epoch_cost.append(cost)
                    if np.isnan(cost):
                        print(str(epoch) + ': Nan error!')
                        self.error_during_train = True
                        return
                    if step == 1 or step % self.decay_steps == 0:
                        avgc = np.mean(epoch_cost)
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tloss: {:.6f} weight:{}'.format(epoch, step, lr, avgc,weight))
                start = start + minlen - 1
                mask = np.arange(len(iters))[(end - start) <= 1]  # 학습 끝난 세션 반환 배치 인덱스 중에서
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(offset_sessions) - 1:  # 마지막 세션 인덱스에 도달 하였을 때 학습 종료
                        finished = True
                        break
                    iters[idx] = maxiter  # 끝난세션 새시작 대치 ex maxiter 49엿는데 세션하나 학습 끝나면 50 으로 바꿈
                    start[idx] = offset_sessions[session_idx_arr[maxiter]]  # 새 세션인덱스 위치 호출
                    end[idx] = offset_sessions[session_idx_arr[maxiter] + 1]  # 새 세션인덱스의 끝 지점 호출
                if len(mask) and self.reset_after_session:  # mask길이가 1이고 reset_after_session이 true 일때 실행
                    for j in range(len(self.attribute_key)):
                        at_states[j][mask] = 0  # 끝난 세션 hidden_state 초기화
            avgc = np.mean(epoch_cost)
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                self.error_during_train = True
                return
            print('Epoch {} \t loss: {:.6f} : '.format(epoch , avgc))
            self.saver.save(self.sess,'{}/weight-model-'.format(self.checkpoint_dir), global_step=epoch)

    def predict_next_batch(self, session_ids, in_idxs, batch=50):
        '''
        :param session_ids: batch_size dim
        :param in_idxs: input array for placeholoder X
        :param batch: batch_size is sames with train batch_size
        :return:
        '''

        if batch != self.batch_size:
            raise Exception('Predict batch size({}) must match train batch size({})'.format(batch, self.batch_size))
        if not self.predict:
            self.current_session = np.ones(batch) * -1
            self.predict = True

        session_change = np.arange(batch)[session_ids != self.current_session]
        if len(session_change) > 0:
            for j in range(len(self.attribute)):
                self.predict_states[j][session_change] = 0.0  # 끝난 세션 히든스테이트 초기화
            self.current_session = session_ids.copy()
        fetcheses = []
        feed_dicts = []
        for i in range(len(self.attribute)):
            fetcheses.append([self.yhats[i], self.final_states[i]])
            feed_dict = {self.Xs[i]: in_idxs[:, i]}

            feed_dict[self.states[i]] = self.predict_states[i]
            feed_dicts.append(feed_dict)
        yhats = []
        for idx, ses in enumerate(self.sesses):
            y, self.predict_states[idx] = ses.run(fetcheses[idx], feed_dict=feed_dicts[idx])
            yhats.append((self.scale(y)+1))
            #yhats.append(y)
        yhats = np.array(yhats)
        preds = self.sess.run(self.yhat2,feed_dict = {self.rating:yhats})

        preds = np.asarray(preds).T
        preds = pd.DataFrame(data=preds ,index=self.item_table[self.item_key])
        return preds



