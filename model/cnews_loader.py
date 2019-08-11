import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

from cws.segmenter import BiLSTMSegmenter
segmenter = BiLSTMSegmenter(data_path='./cws/cws_pos_dict.pkl', model_path='./checkpoints/cws_pos_1107')


def native_word(word, encoding):
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):

    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    contents, labels=[], []

    with open_file(filename) as f:

        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):

    # 构建词汇表
    data_train, _ = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)

    # 对每个字的使用进行计数
    counter = Counter(all_data)
    # 统计出使用最频繁的字
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个<PAD> 来将所有文本pad为同一长度
    words = ['PAD'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):

    # 读取词汇表
    with open_file(vocab_dir) as fp:
        words = [native_content(_.strip()) for _ in fp.readlines()]

    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category(label_path):

    categories = [line.rstrip('\n') for line in open(label_path)]
    categories = [native_content(x) for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    return ''.join(words[x] for x in content)


def process_file2(filename, cat_to_id, max_length=200):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)
    print(cat_to_id)
    data_id, label_id = [], []
    for i in range(len(contents)):
        if labels[i] in cat_to_id:
            # print(segmenter.text2ids(x)[0] for x in contents[i])
            # data_id.append([segmenter.text2ids(x) for x in contents[i]])
            data_id.append(segmenter.text2ids(contents[i])[0])
            label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    # x_pad = segmenter.text2ids(contents)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)
    print(cat_to_id)
    data_id, label_id = [], []
    for i in range(len(contents)):
        if labels[i] in cat_to_id:
            data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
            label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def process_test_file2(filename, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)
    print(cat_to_id)

    data_id, label_id, all_label = [], [], []
    for i in range(len(contents)):
        if labels[i] in cat_to_id:
            # data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
            data_id.append(segmenter.text2ids(contents[i])[0])
            label_id.append(cat_to_id[labels[i]])
            all_label.append(labels[i])
        elif ',' in labels[i]:
            # data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
            data_id.append(segmenter.text2ids(contents[i])[0])
            label = labels[i].split(',')
            label_id.append(cat_to_id[label[0]])
            all_label.append(label)
        elif '，' in labels[i]:
            # data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
            data_id.append(segmenter.text2ids(contents[i])[0])
            label = labels[i].split('，')
            label_id.append(cat_to_id[label[0]])
            all_label.append(label)

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad, all_label


def process_test_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)
    print(cat_to_id)

    data_id, label_id, all_label = [], [], []
    for i in range(len(contents)):
        if labels[i] in cat_to_id:
            data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
            label_id.append(cat_to_id[labels[i]])
            all_label.append(labels[i])
        elif ',' in labels[i]:
            data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
            label = labels[i].split(',')
            label_id.append(cat_to_id[label[0]])
            all_label.append(label)
        elif '，' in labels[i]:
            data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
            label = labels[i].split('，')
            label_id.append(cat_to_id[label[0]])
            all_label.append(label)

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad, all_label


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
