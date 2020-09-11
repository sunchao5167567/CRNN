import tensorflow as tf
import numpy as np
import os
import shutil
from datetime import datetime
import keras.backend as K

def self_Att_channel(x,x_att,r = 16,name = '1'):
    '''
    advanced
    Hu, Jie, Li Shen, and Gang Sun."Squeeze-and-excitation networks." arXiv preprintarXiv:1709.01507 (2017).
    :param x:
    :param r:
    :return:
    '''
    x_self = x
    chanel = K.int_shape(x)[-1]
    L = K.int_shape(x)[-2]

    x_att = tf.keras.layers.GlobalAveragePooling1D(name='self_avg_pool' + name )(x_att)

    # x_att = layers.Conv2D(chanel,
    #                       (H,W),
    #                       padding='valid',
    #                       use_bias=None,
    #                       name='FCN' + name)(x_att)

    x_att = tf.keras.layers.Dense(int(chanel / r),activation='relu')(x_att)
    x_att = tf.keras.layers.Dense(chanel, activation='sigmoid')(x_att)
    x = tf.keras.layers.Multiply()([x_self,x_att])

    return x
	

class SVSRNN(object):

    def __init__(self, num_features, num_rnn_layer = 5, num_hidden_units = [1024, 1024, 1024, 1024 ,1024], tensorboard_directory = 'graphs/svsrnn', clear_tensorboard = True):

        assert len(num_hidden_units) == num_rnn_layer

        self.num_features = num_features
        self.num_rnn_layer = num_rnn_layer
        self.num_hidden_units = num_hidden_units

        self.gstep = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'global_step')
        self.learning_rate = tf.placeholder(tf.float32, shape = [], name = 'learning_rate')

        # The shape of x_mixed, y_src1, y_src2 are [batch_size, n_frames (time), n_frequencies]
        self.x_mixed = tf.placeholder(tf.float32, shape = [None, None, num_features], name = 'x_mixed')  #(64,10,513)
        self.y_src1 = tf.placeholder(tf.float32, shape = [None, None, num_features], name = 'y_src1')
        self.y_src2 = tf.placeholder(tf.float32, shape = [None, None, num_features], name = 'y_src2')

        self.y_pred_src1, self.y_pred_src2 = self.network_initializer()

        # Loss balancing parameter
        self.gamma = 0.05
        self.loss = self.loss_initializer()
        self.optimizer = self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(tensorboard_directory):
            os.makedirs(tensorboard_directory)
        # Tensorboard summary
        if clear_tensorboard:
            shutil.rmtree(tensorboard_directory, ignore_errors = True)
            logdir = tensorboard_directory
        else:
            now = datetime.now()
            logdir = os.path.join(tensorboard_directory, now.strftime('%Y%m%d-%H%M%S'))
        self.writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        self.summary_op = self.summary()

    def network(self):
        # input_layer = self.x_mixed[:,:,:512]
        input_layer = tf.expand_dims(self.x_mixed,-1)  #?*10*513*1
        input_layer = tf.transpose(input_layer, [0, 2, 1, 3])  #?*513*10*1

        # Convolutional Layer 1
        conv1_time = tf.layers.batch_normalization(tf.layers.conv2d(inputs=input_layer, filters=16, kernel_size=[2, 10], strides=[1, 1], padding="same",
                                                               activation=tf.nn.leaky_relu))
        #conv1 = self_Att_channel(x=conv1,x_att=conv1,r=16,name='1')
        # Convolutional Layer 2
        conv1_freq = tf.layers.batch_normalization(tf.layers.conv2d(inputs=input_layer, filters=16, kernel_size=[10, 2],strides=[1, 1], padding="same",
                                                               activation=tf.nn.leaky_relu))
        conv1 = tf.concat([conv1_time,conv1_freq],axis=-1)
        #conv2 = self_Att_channel(x=conv2, x_att=conv2, r=16, name='2')
        # Convolutional Layer 3
        conv2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=conv1, filters=48, kernel_size=[2, 2],strides=[1, 1], padding="same",
                                                               activation=tf.nn.leaky_relu))   #?*513*10*64
        conv2 = self_Att_channel(x=conv2,x_att=conv2,r = 2,name = '1')
        #conv3 = tf.concat([conv1,conv3],axis=-1)
        layer1 = tf.nn.max_pool(conv2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID',name='max_pool')  # ?*256*10*64
        # layer1 = tf.transpose(layer1,[0,2,1,3])  #?*10*(256*64)
        # layer1 = tf.reshape(layer1,[-1,64*256,10])  #(D*Fm)*T  #(?*(64*256)*10)
        #layer1=self_Att_channel(layer1,layer1,r=16,name='5')
        layer1 = tf.transpose(layer1,[0,2,1,3])  #(?*?*256*64)
        # layer1 = attention(layer1,1)
        # fully1_weight = tf.get_variable('fully1_weight', shape=[256 * 64, 1024], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        # fully1_bias = tf.get_variable('fully1_bias', shape=[1024], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        # layer1 = tf.matmul(layer1,)
        # layer_0 = layer1[:,:,:,0]
        # layer_1 = layer1[:,:,:,1]
        # layer_2 = layer1[:,:,:,2]
        # layer = tf.concat([layer_0,layer_1,layer_2],axis=-1)

        #layer1 = self_Att_channel(layer1,x_att=layer1,r=8,name='4')

        layer = layer1[:,:,:,0]
        for i in range(1,48):
            # layer = layer
            layer_ = layer1[:,:,:,i]
            layer = tf.concat([layer,layer_],axis=-1)  #layer1:(?*?*16384)

        layer = tf.concat([layer,self.x_mixed],axis=-1)   #?*?*(16384+513)
		layer = self_Att_channel(layer,x_att=layer,r=2,name='4')

        rnn_layer = [tf.nn.rnn_cell.GRUCell(size) for size in self.num_hidden_units]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layer)
        outputs, state = tf.nn.dynamic_rnn(cell = multi_rnn_cell, inputs = layer, dtype = tf.float32)  #outputs:(?,?,256)  state:(?,256)  inputs:?*10*513
        y_hat_src1 = tf.layers.dense(    #y_hat_src1: shape=(?, 10, 513)
            inputs = outputs,
            units = self.num_features,
            activation = tf.nn.relu,
            name = 'y_hat_src1')
        y_hat_src2 = tf.layers.dense(
            inputs = outputs,
            units = self.num_features,
            activation = tf.nn.relu,
            name = 'y_hat_src2')

        # Time-frequency masking layer时频掩膜
        # np.finfo(float).eps: the smallest representable positive number such that 1.0 + eps != 1.0
        # Absolute value? In principle y_srcs could only be positive in spectrogram
        y_tilde_src1 = y_hat_src1 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed   #y_hat_src1: shape=(?, ?, 513)
        y_tilde_src2 = y_hat_src2 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed
        # Mask with Abs
        #y_tilde_src1 = tf.abs(y_hat_src1) / (tf.abs(y_hat_src1) + tf.abs(y_hat_src2) + np.finfo(float).eps) * self.x_mixed
        #y_tilde_src2 = tf.abs(y_hat_src2) / (tf.abs(y_hat_src1) + tf.abs(y_hat_src2) + np.finfo(float).eps) * self.x_mixed

        return y_tilde_src1, y_tilde_src2
        #return y_hat_src1, y_hat_src2

    def network_initializer(self):

        with tf.variable_scope('rnn_network') as scope:
            y_pred_src1, y_pred_src2 = self.network()

        return y_pred_src1, y_pred_src2


    def generalized_kl_divergence(self, y, y_hat):

        return tf.reduce_mean(y * tf.log(y / y_hat) - y + y_hat)


    def loss_initializer(self):

        with tf.variable_scope('loss') as scope:

            # Mean Squared Error Loss
            # loss = tf.reduce_mean(tf.square(self.y_src1 - self.y_pred_src1) + tf.square(self.y_src2 - self.y_pred_src2), name = 'loss')  #
            # loss = tf.add(
            #     x=self.generalized_kl_divergence(y=self.y_src1, y_hat=self.y_pred_src1),
            #     y=self.generalized_kl_divergence(y=self.y_src2, y_hat=self.y_pred_src2),
            #     name='GKL_loss')
            '''
            # Generalized KL Divergence Loss
            loss = tf.add(
                x = self.generalized_kl_divergence(y = self.y_src1, y_hat = self.y_pred_src1), 
                y = self.generalized_kl_divergence(y = self.y_src2, y_hat = self.y_pred_src2), 
                name = 'GKL_loss')

            # Mean Squared Error + Signal to Inference Ratio Loss
            loss = tf.reduce_mean(tf.square(self.y_src1 - self.y_pred_src1) + tf.square(self.y_src2 - self.y_pred_src2) - self.gamma * (tf.square(self.y_src1 - self.y_pred_src2) + tf.square(self.y_src2 - self.y_pred_src1)), name = 'MSE_SIR_loss')

            # Generalized KL Divergence + Signal to Inference Ratio Loss
            loss = tf.subtract(
                x = (self.generalized_kl_divergence(y = self.y_src1, y_hat = self.y_pred_src1) + self.generalized_kl_divergence(y = self.y_src2, y_hat = self.y_pred_src2)), 
                y = self.gamma * (self.generalized_kl_divergence(y = self.y_src1, y_hat = self.y_pred_src2) + self.generalized_kl_divergence(y = self.y_src2, y_hat = self.y_pred_src1)), 
                name = 'GKL_SIR_loss')
            '''
            loss = tf.reduce_mean(tf.square(self.y_src1 - self.y_pred_src1) + tf.square(self.y_src2 - self.y_pred_src2) - self.gamma * (tf.square(self.y_src1 - self.y_pred_src2) + tf.square(self.y_src2 - self.y_pred_src1)), name = 'MSE_SIR_loss')

        return loss

    def optimizer_initializer(self):

        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss, global_step = self.gstep)

        return optimizer

    def train(self, x, y1, y2, learning_rate):

        #step = self.gstep.eval()

        step = self.sess.run(self.gstep)

        _, train_loss, summaries = self.sess.run([self.optimizer, self.loss, self.summary_op],
            feed_dict = {self.x_mixed: x, self.y_src1: y1, self.y_src2: y2, self.learning_rate: learning_rate})   #x_mixed:(64,10,513)
        self.writer.add_summary(summaries, global_step = step)
        return train_loss

    def validate(self, x, y1, y2):

        y1_pred, y2_pred, validate_loss = self.sess.run([self.y_pred_src1, self.y_pred_src2, self.loss],
            feed_dict = {self.x_mixed: x, self.y_src1: y1, self.y_src2: y2})
        return y1_pred, y2_pred, validate_loss

    def test(self, x):

        y1_pred, y2_pred = self.sess.run([self.y_pred_src1, self.y_pred_src2], feed_dict = {self.x_mixed: x})

        return y1_pred, y2_pred

    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('x_mixed', self.x_mixed)
            tf.summary.histogram('y_src1', self.y_src1)
            tf.summary.histogram('y_src2', self.y_src2)
            summary_op = tf.summary.merge_all()

        return summary_op