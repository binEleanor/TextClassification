# coding: utf-8

import tensorflow as tf
import numpy as np


class TextCNN(object):
    def __init__(self, g_config, cnn_config):
        self.config = g_config
        self.cnn_config = cnn_config
        # 输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.cnn()

    def cnn(self):
        """CNN model"""

        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # 卷积操作
            conv = tf.layers.conv1d(embedding_inputs, self.cnn_config.num_filters, self.cnn_config.kernel_size, name="conv")
            # 池化
            gmp = tf.reduce_max(conv, reduction_indices=[1], name="gmp")

        with tf.name_scope("score"):
            # 池化之后进行全连接操作
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name="fc1")
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            # 再次全连接
            self.logits = tf.layers.dense(fc, self.config.num_classes, name="fc2")
            # 全连接之后进行softmax操作
            self.scores = tf.nn.softmax(self.logits)

            self.y_pred_cls = tf.argmax(self.scores, 1)
            self.y_pred_proba = tf.reduce_max(self.scores, 1)  # 取最大类别的概率

        with tf.name_scope("optimize"):
            # 计算交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            # self.optim = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


class TextRNN_Attention(object):
    """
        RNN with Attention mechanism for text classification
        """

    def __init__(self, g_config, rnn_config):
        self.vocab_size = g_config.vocab_size
        self.embedding_size = g_config.embedding_dim
        self.num_classes = g_config.num_classes
        self.sequence_length = g_config.seq_length
        self.rnn_size = rnn_config.rnn_size
        self.num_layers = rnn_config.num_layers
        self.attention_size = rnn_config.attention_size
        self.learning_rate = g_config.learning_rate

        # 输入input_x shape[?, sequence_length], input_y shape[?, num_classes]
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='input_y')
        # self.seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # 基本的循环神经网络模块　GRU cell
        def basic_rnn_cell(rnn_size):
            return tf.contrib.rnn.GRUCell(rnn_size)
            # return tf.contrib.rnn.LSTMCell(rnn_size)

        # 前向 RNN model
        with tf.name_scope('fw_rnn'):
            fw_rnn_cell = tf.contrib.rnn.MultiRNNCell([basic_rnn_cell(self.rnn_size) for _ in range(self.num_layers)])
            fw_rnn_cell = tf.contrib.rnn.DropoutWrapper(fw_rnn_cell, output_keep_prob=self.keep_prob)

        # 反向 RNN model
        with tf.name_scope('bw_rnn'):
            bw_rnn_cell = tf.contrib.rnn.MultiRNNCell([basic_rnn_cell(self.rnn_size) for _ in range(self.num_layers)])
            bw_rnn_cell = tf.contrib.rnn.DropoutWrapper(bw_rnn_cell, output_keep_prob=self.keep_prob)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), trainable=True,
                                         name='W')
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope('bi_rnn'):
            rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, inputs=embedding_inputs,
                                                            dtype=tf.float32)
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs
        if isinstance(rnn_output, tuple):
            rnn_output = tf.concat(rnn_output, 2)

        # Attention Layer
        with tf.name_scope('attention'):
            input_shape = rnn_output.shape        # (batch_size, sequence_length, hidden_size)
            sequence_size = input_shape[1].value  # the length of sequences processed in the RNN layer
            hidden_size = input_shape[2].value    # hidden size of the RNN layer

            # 定义注意力机制的变量
            attention_w = tf.Variable(tf.truncated_normal([hidden_size, self.attention_size], stddev=0.1),
                                      name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[self.attention_size]), name='attention_b')
            attention_u = tf.Variable(tf.truncated_normal([self.attention_size], stddev=0.1), name='attention_u')

            z_list = []

            for t in range(sequence_size):
                u_t = tf.tanh(tf.matmul(rnn_output[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
                z_t = tf.matmul(u_t, tf.reshape(attention_u, [-1, 1]))
                z_list.append(z_t)

            # Transform to batch_size * sequence_size
            # 得到每个词的权重
            attention_z = tf.concat(z_list, axis=1)
            # 得到每个词的attention概率
            self.alpha = tf.nn.softmax(attention_z)
            # Transform to batch_size * sequence_size * 1 , same rank as rnn_output
            # 对sequence的run_output与attention权重进行求和，求和之后作为attention的输出
            attention_output = tf.reduce_sum(rnn_output * tf.reshape(self.alpha, [-1, sequence_size, 1]), 1)

        # Add dropout
        with tf.name_scope('dropout'):
            # attention_output shape: (batch_size, hidden_size)
            self.final_output = tf.nn.dropout(attention_output, self.keep_prob)

        # Fully connected layer
        with tf.name_scope('output'):
            fc_w = tf.Variable(tf.truncated_normal([hidden_size, self.num_classes], stddev=0.1), name='fc_w')
            fc_b = tf.Variable(tf.zeros([self.num_classes]), name='fc_b')
            self.logits = tf.matmul(self.final_output, fc_w) + fc_b
            self.y_pred_cls = tf.argmax(self.logits, 1, name='predictions')

        # Calculate cross-entropy loss
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        # Create optimizer
        with tf.name_scope('optimization'):
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            # optimizer = tf.train.AdamOptimizer(learning_rate)
            # gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            # gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
            # self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        # Calculate accuracy
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.y_pred_cls, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


class TextCNN_Attention(object):
    def __init__(self, g_config, cattnn_config):
        self.config = g_config
        self.cattnn_config = cattnn_config
        # 输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.cnn_attention()

    def cnn_attention(self):

        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # 卷积之后的向量维度为 [batch_size的大小, 句子长度－kernel_size+1, num_filters]
            conv = tf.layers.conv1d(embedding_inputs, self.cattnn_config.num_filters, self.cattnn_config.kernel_size, name="conv")

        # Attention Layer
        with tf.name_scope('attention'):
            input_shape = conv.shape              # (batch_size, sequence_length-kernel_size+1, num_filters)
            sequence_size = input_shape[1].value  # the length of sequences processed in the CNN layer
            hidden_size = input_shape[2].value    # num_filters size of the CNN layer

            attention_w = tf.Variable(tf.truncated_normal([hidden_size, self.cattnn_config.attention_size], stddev=0.1),
                                      name='attention_w')  # shape [hidden_size, self.config.attention_size]
            attention_b = tf.Variable(tf.constant(0.1, shape=[self.cattnn_config.attention_size]), name='attention_b')
            attention_u = tf.Variable(tf.truncated_normal([self.cattnn_config.attention_size], stddev=0.1), name='attention_u')
            z_list = []
            # 计算每个单词的权重系数
            for t in range(sequence_size):
                # Location-based Attention
                u_t = tf.tanh(tf.matmul(conv[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
                z_t = tf.matmul(u_t, tf.reshape(attention_u, [-1, 1]))
                z_list.append(z_t)

            attention_z = tf.concat(z_list, axis=1)                    # shape[batch_size, sequence_size-kernel_size+1]
            # 获取每个单词的　attention 值
            self.alpha = tf.nn.softmax(attention_z)                    # shape[batch_size, sequence_size-kernel_size+1]

            gmp = tf.reduce_max(conv * tf.reshape(self.alpha, [-1, sequence_size, 1]),
                                reduction_indices=[1], name="gmp")     # shape [batch_size, sequence_size]

        with tf.name_scope("score"):
            # 池化之后进行全连接操作
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name="fc1")
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 再次全连接
            self.logits = tf.layers.dense(fc, self.config.num_classes, name="fc2")
            # 全连接之后进行softmax操作
            self.scores = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(self.scores, 1)
            self.y_pred_proba = tf.reduce_max(self.scores, 1)  # 取最大类别的概率

        with tf.name_scope("optimize"):
            # 计算交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
