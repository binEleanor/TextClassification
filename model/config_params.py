# coding: utf-8

import tensorflow as tf
import numpy as np


class Global_Config(object):

    seq_length = 200
    embedding_dim = 256
    num_classes = 8
    # 词汇表大小
    vocab_size = 5000
    # 全连接神经元
    hidden_dim = 128
    dropout_keep_prob = 0.5
    learning_rate = 1e-3
    # 每批训练大小
    batch_size = 64
    num_epochs = 10
    print_per_batch = 100
    save_per_batch = 10


class CNN_Config(object):
    # 卷积神经网络参数
    num_filters = 256
    kernel_size = 4


class CNN_Attention_Config(object):
    # 卷积神经网络参数
    num_filters = 256
    kernel_size = 4
    attention_size = 50


class RNN_Attention_Config(object):
    # 循环神经网络+Attention机制参数
    num_filters = 256
    kernel_size = 4
    rnn_size = 100
    num_layers = 2
    attention_size = 50