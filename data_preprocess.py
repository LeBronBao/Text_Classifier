# -*- encoding: utf-8 -*-

from util import load_word_ebd, split_train_test_for_yelp, split_train_test_for_text_etm, rm_useless_tokens, get_all_sentences
from util import read_dbpedia_data, read_imdb_data, represent_food_review_with_kw
from util import split_train_test_for_amazon, split_yelp_train_data_to_cats, represent_review_with_kw, split_imdb_train_data_to_cats
from util import split_train_test_for_mobile, split_train_test_for_db_amazon, load_neg_words
from lda_model import model_lda_to_words, model_lda_to_dicts, model_lda_for_imdb_cats, model_lda_for_all, lda_match
from lda_model import filter_lda_cats_by_center_word, filter_lda_cats_by_mean_word_embed
from lda_model import get_similar_lda_cat
from eda import eda
from seq_seg import seg_sequences
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import random

neg_words_path = 'yelp_glove/negative_words'


# 将yelp数据集训练、验证、测试集以及词嵌入转化为网络所需的输入形式
def preprocess_yelp_clf_data(model_path, review_paths, max_len, dim):
    # 读取预训练词向量
    word_embed_dict = load_word_ebd(model_path, loading_num=1000000)
    # 划分训练、验证、测试集，该步中只移除了无用的标点，注意此步骤中并未去除停用词
    train_x, train_y, test_x, test_y = split_train_test_for_yelp(review_paths)

    # 将测试、验证、训练集的所有句子合并，用于下一步统计词语数量
    sentences = get_all_sentences(train_x, test_x)

    # 统计词语总数
    print("Begin to transform text to int nd-array...")
    t = Tokenizer()
    t.fit_on_texts(sentences)
    vocab_size = len(t.word_index) + 1

    # 将原始句子文本转化为整数token表示形式，此处若是与LDA一同输入，则需要使用 train_ori_x 而非 train_x
    encoded_train_docs = t.texts_to_sequences(train_x)
    encoded_test_docs = t.texts_to_sequences(test_x)

    # 将上述原始句子的整数token表示形式转化为 nd-array
    padded_train_docs = pad_sequences(encoded_train_docs, maxlen=max_len, padding='post')
    padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_len, padding='post')

    # 将出现在样本中的词语以及对应词向量加入字典
    embedding_matrix = np.zeros((vocab_size, dim))
    count = 0
    for word, i in t.word_index.items():
        try:
            vec = word_embed_dict[word]
            embedding_matrix[i] = vec
            count += 1
            print("Constructing embedding matrix.Finish:" + str(count))
        except KeyError:
            continue

    dic = {}
    dic['vocab_size'] = vocab_size
    dic['train_x'] = padded_train_docs
    dic['train_y'] = train_y
    dic['test_x'] = padded_test_docs
    dic['test_y'] = test_y
    dic['embed'] = embedding_matrix

    return dic


# 将IMDB影评数据划分为训练、验证、测试集以及词嵌入转化为网络所需的输入形式
def preprocess_imdb_clf_data(model_path, imdb_train_path, imdb_test_path, max_len, kw_max_len, dim):
    # 读取预训练词向量
    word_embed_dict = load_word_ebd(model_path, loading_num=1000000)

    train_x, train_y = read_imdb_data(imdb_train_path)
    test_x, test_y = read_imdb_data(imdb_test_path)

    train_x = train_x[:10000]
    train_y = train_y[:10000]

    test_x = test_x[:2500]
    test_y = test_y[:2500]

    sentences = get_all_sentences(train_x, test_x)

    print("Begin to transform text into int nd-array...")
    t = Tokenizer()
    t.fit_on_texts(sentences)
    vocab_size = len(t.word_index) + 1

    # 将原始句子文本转化为整数token表示形式，此处若是与LDA一同输入，则需要使用 train_ori_x 而非 train_x
    encoded_train_docs = t.texts_to_sequences(train_x)
    encoded_test_docs = t.texts_to_sequences(test_x)

    # 将上述原始句子的整数token表示形式转化为 nd-array
    padded_train_docs = pad_sequences(encoded_train_docs, maxlen=max_len, padding='post')
    padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_len, padding='post')

    # 将出现在样本中的词语以及对应词向量加入字典
    embedding_matrix = np.zeros((vocab_size, dim))
    count = 0
    for word, i in t.word_index.items():
        try:
            vec = word_embed_dict[word]
            embedding_matrix[i] = vec
            count += 1
            print("Constructing embedding matrix. Finish:" + str(count))
        except KeyError:
            continue

    dic = {}
    dic['vocab_size'] = vocab_size
    dic['train_x'] = padded_train_docs
    dic['train_y'] = train_y
    dic['test_x'] = padded_test_docs
    dic['test_y'] = test_y
    dic['embed'] = embedding_matrix

    return dic


# 将Amazon食品评论数据划分为训练、验证、测试集，并转化为网络所需的输入形式
def preprocess_amazon_clf_data(model_path, reviews_path, max_len, dim):
    # 读取预训练词向量
    word_embed_dict = load_word_ebd(model_path, loading_num=1000000)
    # 划分训练、验证、测试集，该步中只移除了无用的标点，注意此步骤中并未去除停用词
    train_x, train_y, test_x, test_y = split_train_test_for_db_amazon(reviews_path)

    # 将测试、验证、训练集的所有句子合并，用于下一步统计词语数量
    sentences = get_all_sentences(train_x, test_x)
    seg_sequences(word_embed_dict, sentences)

    # 统计词语总数
    print("Begin to transform text to int nd-array...")
    t = Tokenizer()
    t.fit_on_texts(sentences)
    vocab_size = len(t.word_index) + 1

    # 将原始句子文本转化为整数token表示形式，此处若是与LDA一同输入，则需要使用 train_ori_x 而非 train_x
    encoded_train_docs = t.texts_to_sequences(train_x)
    encoded_test_docs = t.texts_to_sequences(test_x)

    # 将上述原始句子的整数token表示形式转化为 nd-array
    padded_train_docs = pad_sequences(encoded_train_docs, maxlen=max_len, padding='post')
    padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_len, padding='post')

    # 将出现在样本中的词语以及对应词向量加入字典
    embedding_matrix = np.zeros((vocab_size, dim))
    count = 0
    for word, i in t.word_index.items():
        try:
            vec = word_embed_dict[word]
            embedding_matrix[i] = vec
            count += 1
            print("Constructing embedding matrix.Finish:" + str(count))
        except KeyError:
            continue

    dic = {}
    dic['vocab_size'] = vocab_size

    dic['train_x'] = padded_train_docs
    dic['train_y'] = train_y

    dic['test_x'] = padded_test_docs
    dic['test_y'] = test_y

    dic['embed'] = embedding_matrix

    return dic


# Amazon电子设备评论
def preprocess_amazon_mobile_data(model_path, review_paths, max_len, dim):
    # 读取预训练词向量
    eg_x, eg_y = load_eg()
    word_embed_dict = load_word_ebd(model_path, loading_num=1000000)
    # 划分训练、验证、测试集，该步中只移除了无用的标点，注意此步骤中并未去除停用词
    train_x, train_y, test_x, test_y, dev_x, dev_y = split_train_test_for_mobile(review_paths)
    # 后续会将训练集当做参数传入处理而改变其原始内容，提前将内容赋值

    # 将测试、验证、训练集的所有句子合并，用于下一步统计词语数量
    train_x = seg_sequences(word_embed_dict, train_x)
    test_x = seg_sequences(word_embed_dict, test_x)
    sentences = get_all_sentences(train_x, test_x, dev_x)

    # 统计词语总数
    print("Begin to transform text to int nd-array...")
    t = Tokenizer()
    t.fit_on_texts(sentences)
    vocab_size = len(t.word_index) + 1

    # 将原始句子文本转化为整数token表示形式，此处若是与LDA一同输入，则需要使用 train_ori_x 而非 train_x
    encoded_train_docs = t.texts_to_sequences(train_x)
    encoded_test_docs = t.texts_to_sequences(test_x)
    encoded_eg_docs = t.texts_to_sequences(eg_x)

    # 将上述原始句子的整数token表示形式转化为 nd-array
    padded_train_docs = pad_sequences(encoded_train_docs, maxlen=max_len, padding='post')
    padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_len, padding='post')
    padded_eg_docs = pad_sequences(encoded_eg_docs, maxlen=max_len, padding='post')

    # 将出现在样本中的词语以及对应词向量加入字典
    embedding_matrix = np.zeros((vocab_size, dim))
    count = 0
    for word, i in t.word_index.items():
        try:
            vec = word_embed_dict[word]
            embedding_matrix[i] = vec
            count += 1
            print("Constructing embedding matrix.Finish:" + str(count))
        except KeyError:
            continue

    dic = {}
    dic['vocab_size'] = vocab_size

    dic['train_x'] = padded_train_docs
    dic['train_y'] = train_y
    dic['test_x'] = padded_test_docs
    dic['test_y'] = test_y
    dic['eg_x'] = padded_eg_docs
    dic['eg_y'] = eg_y

    dic['embed'] = embedding_matrix

    return dic


# DBPedia主题
def preprocess_dbpedia_data(model_path, review_paths, max_len, dim):
    # 读取预训练词向量
    word_embed_dict = load_word_ebd(model_path, loading_num=1000000)

    train_x, train_y = read_dbpedia_data(review_paths[0])
    test_x, test_y = read_dbpedia_data(review_paths[1])

    sentences = get_all_sentences(train_x, test_x)

    print("Begin to transform text into int nd-array...")
    t = Tokenizer()
    t.fit_on_texts(sentences)
    vocab_size = len(t.word_index) + 1

    # 将原始句子文本转化为整数token表示形式，此处若是与LDA一同输入，则需要使用 train_ori_x 而非 train_x
    encoded_train_docs = t.texts_to_sequences(train_x)
    encoded_test_docs = t.texts_to_sequences(test_x)

    # 将上述原始句子的整数token表示形式转化为 nd-array
    padded_train_docs = pad_sequences(encoded_train_docs, maxlen=max_len, padding='post')
    padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_len, padding='post')

    # 将出现在样本中的词语以及对应词向量加入字典
    embedding_matrix = np.zeros((vocab_size, dim))
    count = 0
    for word, i in t.word_index.items():
        try:
            vec = word_embed_dict[word]
            embedding_matrix[i] = vec
            count += 1
            print("Constructing embedding matrix. Finish:" + str(count))
        except KeyError:
            continue

    dic = {}
    dic['vocab_size'] = vocab_size
    dic['train_x'] = padded_train_docs
    dic['train_y'] = train_y
    dic['test_x'] = padded_test_docs
    dic['test_y'] = test_y
    dic['embed'] = embedding_matrix

    return dic


# 将IMDB训练和测试数据分别读进两个文件中
def read_imdb_to_one_file():
    test_neg_path = 'Imdb/aclImdb/test/neg'
    test_pos_path = 'Imdb/aclImdb/test/pos'
    train_neg_path = 'Imdb/aclImdb/train/neg'
    train_pos_path = 'Imdb/aclImdb/train/pos'

    test_target_path = 'Imdb/test_data'
    train_target_path = "Imdb/train_data"

    test_neg_files = os.listdir(test_neg_path)
    test_pos_files = os.listdir(test_pos_path)
    train_neg_files = os.listdir(train_neg_path)
    train_pos_files = os.listdir(train_pos_path)

    test_reviews = []
    train_reviews = []
    i = 0
    for file in test_neg_files:
        with open(test_neg_path+'/'+file, 'r', encoding='utf-8') as f:
            rating = file.split('_')[1].replace('.txt', '')
            test_reviews.append(f.readlines()[0]+" "+rating+'\n')
        print(i)
        i += 1
    for file in test_pos_files:
        with open(test_pos_path+'/'+file, 'r', encoding='utf-8') as f:
            rating = file.split('_')[1].replace('.txt', '')
            test_reviews.append(f.readlines()[0]+" "+rating+'\n')
        print(i)
        i += 1

    for file in train_neg_files:
        with open(train_neg_path+'/'+file, 'r', encoding='utf-8') as f:
            rating = file.split('_')[1].replace('.txt', '')
            train_reviews.append(f.readlines()[0]+' '+rating+'\n')
        print(i)
        i += 1
    for file in train_pos_files:
        with open(train_pos_path+'/'+file, 'r', encoding='utf-8') as f:
            rating = file.split('_')[1].replace('.txt', '')
            train_reviews.append(f.readlines()[0]+' '+rating+'\n')
        print(i)
        i += 1

    random.shuffle(test_reviews)
    random.shuffle(train_reviews)

    with open(test_target_path, 'w', encoding='utf-8') as f:
        for review in test_reviews:
            f.write(review)
    with open(train_target_path, 'w', encoding='utf-8') as f:
        for review in train_reviews:
            f.write(review)

    print()


def load_eg():
    list = []
    labels = []
    with open('debug_data/amazon_mobile_data', 'r', encoding='utf-8') as f:
        for line in f:
            if ' 1\n' in line:
                labels.append(1)
            elif ' 2\n' in line:
                labels.append(2)
            elif ' 3\n' in line:
                labels.append(3)
            elif ' 4\n' in line:
                labels.append(4)
            elif ' 5\n' in line:
                labels.append(5)
            new_sent = rm_useless_tokens(line[:-2].lower())
            list.append(new_sent)

    return list, labels

