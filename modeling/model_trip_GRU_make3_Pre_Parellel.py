"""
Author : Lee Won Seok (reference 'GRU4REC_Tensorflow' ,Author : Weiping Song)
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from random import shuffle
from .gru import GRU


class GRU4Rec():
    def __init__(self, sess, args):

        self.sess = sess
        self.is_trainging = args.is_training
        self.layers = args.layers
        self.rnn_size = args.rnn_size
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.dropout_p_hidden = args.dropout_p_hidden
        self.learning_rate = args.learning_rate
        self.decay = args.decay
        self.decay_steps = args.decay_steps
        self.initializer = args.initializer
        self.reset_after_session = args.reset_after_session
        self.session_key = args.session_key
        self.attribute_key = args.attribute_key
        self.time_key = args.time_key
        self.item_key = args.item_key
        self.grad_cap = args.grad_cap
        self.attribute = args.attribute
        self.max_to_keep = args.max_ckpt_keep
        self.predict_state = np.zeros([self.batch_size,self.rnn_size], dtype=np.float32)
        self.item_table = args.item_table
        self.init_as_normal = args.init_as_normal
        self.test_model = args.test_model
        self.retraining = args.retraining

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

        self.checkpoint_dir = args.checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            print('Checkpoint Dir not found I make it')
            os.makedirs(self.checkpoint_dir)

        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.max_to_keep)

        if self.is_trainging & self.retraining:
            return  # 훈련중이면 여기서 멈추게

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, '{}/attribute_model_{}-{}'.format(self.checkpoint_dir,self.attribute_key,self.test_model))

        ##########################ACTIVATE FUNCTION###################################3333

    def linear(self, X):
        return X

    def tanh(self, X):
        return tf.nn.tanh(X)

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

    def build_model(self):
        batch_size = self.batch_size
        self.keep_prob = tf.placeholder(tf.float32, name="d_out")
        self.X = tf.placeholder(tf.int32, [batch_size], name='input')
        self.Y = tf.placeholder(tf.int32, [batch_size], name='output')

        self.state = tf.placeholder(tf.float32, [batch_size, self.rnn_size], name='rnn_state')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('gru_layer'):
            sigma = np.sqrt(6.0)/(self.rnn_size + self.attribute)

            if self.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
            embedding = tf.get_variable("embedding_"+ self.attribute_key,[self.attribute,self.rnn_size],initializer=initializer)
            softmax_W = tf.get_variable("softmax_w_"+ self.attribute_key,[self.attribute,self.rnn_size],initializer=initializer)
            softmax_b = tf.get_variable("softmax_b_"+ self.attribute_key,[self.attribute],initializer=tf.zeros_initializer())


            cell = GRU(input_dim=self.rnn_size, hidden_size=self.rnn_size, dropout=True,
                       dropout_p_hiddden=self.dropout_p_hidden)


            inputs = tf.nn.embedding_lookup(embedding, self.X)

            outputs, state = cell.forward_process(inputs, self.state)

            print(outputs.shape)
            # outputs = tf.contrib.layers.batch_norm(outputs,
            #                                    center=True, scale=True,
            #                                    is_training=self.is_trainging)
            # state = tf.contrib.layers.batch_norm(state,
            #                                    center=True, scale=True,
            #                                    is_training=self.is_trainging)
            self.final_state = state

        if self.is_trainging:
            '''
            Use other examples of the minibatch as negative samples.
            '''
            sampled_W = tf.nn.embedding_lookup(softmax_W, self.Y) # y 배치 뽑기
            sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y)
            logits = tf.matmul(outputs, sampled_W, transpose_b=True) + sampled_b # batch_size x batch_size
            self.yhat = self.final_activation(logits)
            self.cost = self.loss_function(self.yhat)

        else:
            logits = tf.matmul(outputs, softmax_W, transpose_b=True) + softmax_b # (batch_sizexrnn) x (rnn_Size x n_item\1\5)
            self.yhat = self.final_activation(logits)

        if not self.is_trainging:
            return

        self.lr = tf.maximum(1e-5, tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                              self.decay, staircase=True))
        '''
        Optimizer
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)

        tvars = tf.trainable_variables()
        gvs = optimizer.compute_gradients(self.cost, tvars)
        if self.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in gvs]
        else:
            capped_gvs = gvs
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

    def init(self, data):
        uni = data[self.session_key].unique()  # 추가
        shuffle(uni)  # 추가
        new = pd.DataFrame(uni, columns=[self.session_key])  # 추가
        new['sort'] = range(0, uni.shape[0])  # 추가
        data2 = new.merge(data, how='inner', on=self.session_key)  # 추가
        data2.sort_values(['sort', self.time_key], inplace=True)
        offset_sessions = np.zeros(data2['sort'].nunique() + 1, dtype=np.int32)
        offset_sessions[1:] = data2.groupby('sort').size().cumsum()  # session key기준으로 누적합계 구하기
        return offset_sessions, data2

    def fit(self, data2):
        '''
        item_table frame in columns
        item_id attribute1 attribute2 ....
        '''
        self.error_during_train = False  # ?? 어디에 쓰이노
        print('fitting model....')
        for epoch in list(range(self.n_epochs)):
            offset_sessions, data = self.init(data2)
            epoch_cost = []

            state = np.zeros([self.batch_size, self.rnn_size], dtype=np.float32)

            session_idx_arr = np.arange(len(offset_sessions) - 1)  # session index 담긴 어레이
            iters = np.arange(self.batch_size)  # 0~batchsize-1 까지
            maxiter = iters.max()

            start = offset_sessions[session_idx_arr[iters]]  # 0번 부터 49번 까지 start 에 담고 데이터안에서 각 세션 시작 인덱스를 담고있음
            end = offset_sessions[session_idx_arr[iters] + 1]  # 1번 부터 50번 까지 end에 담음 앞에 각 세션이 끝나는 시점의 인덱스

            finished = False

            while not finished:

                minlen = (end - start).min()  # 전체 세션중 최소길이#
                out_idx = data[self.attribute_key].values[start]
                # 피보나치!!
                for i in range(minlen - 1):  # 첫 값은 out으로 이용되지 않음
                    in_idx = out_idx
                    out_idx = data[self.attribute_key].values[start + 1]

                    fetches = [self.cost, self.final_state, self.global_step, self.lr, self.train_op]
                    feed_dict = {self.X: in_idx, self.Y: out_idx, self.keep_prob: self.dropout_p_hidden,
                                 self.state: state}

                    cost, state, step, lr, _ = self.sess.run(fetches, feed_dict=feed_dict)
                    epoch_cost.append(cost)

                    if np.isnan(cost):
                        print(str(epoch) + ': Nan error!')
                        self.error_during_train = True
                        return
                    if step == 1 or step % self.decay_steps == 0:
                        avgc = np.mean(epoch_cost)
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tloss: {:.6f}'.format(epoch, step, lr, avgc))

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
                    for i in list(range(self.layers)):
                        state[i][mask] = 0  # 끝난 세션 hidden_state 초기화
            avgc = np.mean(epoch_cost)
            print('Epoch: {} cost: {}'.format(epoch, avgc))
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                self.error_during_train = True
                return
            if self.retraining == False:
                self.saver.save(self.sess, '{}/attribute_model_{}'.format(self.checkpoint_dir,self.attribute_key),
                                global_step=epoch + self.test_model + 1)
            else:
                self.saver.save(self.sess, '{}/attribute_model_{}'.format(self.checkpoint_dir,self.attribute_key), global_step=epoch)

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
            for i in range(self.layers):
                self.predict_state[i][session_change] = 0.0  # 끝난 세션 히든스테이트 초기화
            self.current_session = session_ids.copy()
        fetches = [self.yhat, self.final_state]
        feed_dict = {self.X: in_idxs, self.keep_prob: self.dropout_p_hidden}
        feed_dict[self.state] = self.predict_state
        preds, self.predict_state = self.sess.run(fetches, feed_dict)  # batch_size x n_items

        preds = np.asarray(preds).T

        preds = pd.DataFrame(data=preds)

        return preds