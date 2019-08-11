#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
import datetime
import tensorflow as tf
import sklearn
import numpy as np
import pandas as pd

from cnews_loader import *
from config_params import *
from model import TextCNN, TextRNN_Attention, TextCNN_Attention

base_dir = './train'
train_dir = os.path.join(base_dir, 'text.train.txt')
test_dir = os.path.join(base_dir, 'test_wit hout.txt')
val_dir = os.path.join(base_dir, 'text.val.txt')
vocab_dir = os.path.join(base_dir, 'text.vocab.txt')
label_path = os.path.join(base_dir, 'labels.txt')


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time

    return datetime.timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch,  keep_prob, model):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        # model.seq_len: seq_len,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_, model):

    # 评估在某一数据上的准确率和损失
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0

    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0, model)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def evaluate_test(sess, x_, y_, all_label):
    # 评估在某一数据上的准确率和损失
    data_len = len(x_)

    right = 0
    for i in range(len(x_)):
        if y_[i] in all_label[i]:
            right += 1
    return right / data_len


def train(save_dir, cat_to_id, word_to_id, model, config):
    print("Configuring TensorBoard and Saver ...")
    # 配置tensotboard
    tensorboard_dir = 'tensorboard/text_rnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)

    # merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'best_validation')
    print("Loading training and validation data")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)

    time_dif = get_time_dif(start_time)
    print("Time Usage:", time_dif)

    # 创建Session()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    print("Training and evaluating")
    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 1000

    flag = False
    for epoch in range(config.num_epochs):
        print("Epoch: ", epoch + 1)

        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob, model)
            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将结果写入tensorboard
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)
            if total_batch % config.print_per_batch == 0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = sess.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(sess, x_val, y_val, model)
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=sess, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss:{1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss:{3:>6.2}, Val Acc: {4:>7.2}, Time:{5}{6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            sess.run(model.optim, feed_dict=feed_dict)
            total_batch += 1
            if total_batch - last_improved > require_improvement:
                # 验证集
                print("No optimization for a long time, auto stopping ...")
                flag = True
                break
        if flag:
            break


def test(save_dir, cat_to_id, word_to_id, model, g_config, categories):
    print("Loading test data...")
    start_time = time.time()

    x_test, y_test, all_label = process_test_file(test_dir, word_to_id, cat_to_id, g_config.seq_length)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    # 读取保存的模型
    save_path = os.path.join(save_dir, 'best_validation')
    saver.restore(sess=sess, save_path=save_path)

    print("Testing...")
    loss_test, acc_test = evaluate(sess, x_test, y_test, model)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)

    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id: end_id],
            model.keep_prob: 1.0
        }

        y_pred_cls[start_id: end_id] = sess.run(model.y_pred_cls, feed_dict=feed_dict)

    contents, origin_label = [], []
    with open(test_dir, 'r') as f:
        for line in f:
            contents.append(line.strip().split('\t')[1])
            origin_label.append(line.strip().split('\t')[0])

    id_to_cat = {value1: key1 for key1, value1 in cat_to_id.items()}
    y_pred = [id_to_cat.get(item) for item in y_pred_cls]

    acc_test = evaluate_test(sess, x_test, y_pred, all_label)
    msg = 'Test Acc: {0:>6.2}'
    print(msg.format(acc_test))

    # res = pd.DataFrame({'contents': contents, 'origin_label': origin_label, 'predict_label': y_pred})
    # res.to_csv('./predict/pred_rnn_test_nolabel.csv')

    # 评估
    print("Precision, Recall and F1-Score....")
    print(sklearn.metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = sklearn.metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time Usage:", time_dif)


def run_cnn(isTrain):
    print("Configuring CNN model")
    g_config = Global_Config()
    cnn_config = CNN_Config()
    save_dir = './checkpoint/text_cnn_model'

    if not os.path.exists(vocab_dir):
        build_vocab(train_dir, vocab_dir, g_config.vocab_size)

    categories, cat_to_id = read_category(label_path)
    words, word_to_id = read_vocab(vocab_dir)

    g_config.vocab_size = len(words)
    model = TextCNN(g_config, cnn_config)
    if isTrain:
        train(save_dir, cat_to_id, word_to_id, model, g_config)
    else:
        test(save_dir, cat_to_id, word_to_id, model, g_config, categories)


def run_cnn_attention(isTrain):
    print("Configuring CANN model")
    save_dir = './checkpoint/text_catnn_model'
    g_config = Global_Config()
    cattnn_config = CNN_Attention_Config()

    if not os.path.exists(vocab_dir):
        build_vocab(train_dir, vocab_dir, g_config.vocab_size)

    categories, cat_to_id = read_category(label_path)
    words, word_to_id = read_vocab(vocab_dir)

    g_config.vocab_size = len(words)
    model = TextCNN_Attention(g_config, cattnn_config)
    if isTrain:
        train(save_dir, cat_to_id, word_to_id, model, g_config)
    else:
        test(save_dir, cat_to_id, word_to_id, model, g_config, categories)


def run_rnn(isTrain):
    print("Configuring RNN model")
    g_config = Global_Config()
    rnn_config = RNN_Attention_Config()

    save_dir = './checkpoint/text_rnn_model'
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir, vocab_dir, g_config.vocab_size)
    categories, cat_to_id = read_category(label_path)
    words, word_to_id = read_vocab(vocab_dir)
    g_config.vocab_size = len(words)
    model = TextRNN_Attention(g_config, rnn_config)
    if isTrain:
        train(save_dir, cat_to_id, word_to_id, model, g_config)
    else:
        test(save_dir, cat_to_id, word_to_id, model, g_config, categories)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['cnn', 'rnn', 'cnn_attention']:
        raise ValueError("usage: python run_rnn.py [cnn / rnn / cnn_attention]")

    print("Configuring Model")

    # 表示为训练或者测试，True表示模型训练
    isTrain = False

    if sys.argv[1] == 'cnn':
        run_cnn(isTrain)
    elif sys.argv[1] == 'rnn':
        run_rnn(isTrain)
    else:
        run_cnn_attention(isTrain)
