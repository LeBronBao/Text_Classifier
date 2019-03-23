# -*- encoding: utf-8 -*-


#import keras.backend.tensorflow_backend as KTF
import os
from keras.layers.normalization import BatchNormalization
#from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout, Flatten, K
from keras.layers import LSTM, Bidirectional, concatenate, Input, Embedding, Add, Multiply, Dot, GRU, Average
from keras.layers import Conv1D, GlobalMaxPool1D, GlobalAveragePooling1D, Lambda, Permute, Activation
from keras.activations import softmax, relu
from attention import Attention
from gensim.models import word2vec
from data_preprocess import preprocess_yelp_clf_data, preprocess_imdb_clf_data, preprocess_dbpedia_data
from data_preprocess import preprocess_amazon_clf_data, preprocess_amazon_mobile_data
from transformer import MyTransformerEncoder

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


train_file_path = "training_data/seg_sent_without_label"
word2vec_file_path = "output_result/up_down_stream.model"  # 词嵌入模型路径
vector_file_path = "output_result/up_down_stream.vector"  # 词嵌入向量路径
text_file_path = "training_data/seg_sent_with_label"  # 带标注的文本文件路路径

train_text_path = "train_test_data/train_text"  # 训练集句子
test_text_path = "train_test_data/test_text"  # 测试集句子
predicted_test_path = "train_test_data/predicted_test_text"  # 标签为被LSTM预测的句子


# 训练词向量模型
def train_word2vec():
    sentences = word2vec.Text8Corpus(train_file_path)
    model = word2vec.Word2Vec(sentences, sg=1, min_count=5, window=5, )

    model.save(word2vec_file_path)
    model.wv.save_word2vec_format(vector_file_path, binary=False)

    print("Finish training. Model saved!")
    sim_word = model.most_similar(["医院"], topn=20)
    for word in sim_word:
        for w in word:
            print(w)


# Yelp评论
def model_yelp_reviews(model_path, review_paths, max_len, dim):
    data_dict = preprocess_yelp_clf_data(model_path, review_paths, max_len, dim)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']
    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    input = Input(shape=(max_len,), dtype='int32')  # 原始文本
    embed = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input)
    embed_dp = Dropout(0.3)(embed)

    conv1 = Conv1D(filters=dim, kernel_size=3, padding='same', activation='relu', strides=1)(embed_dp)
    conv2 = Conv1D(filters=dim, kernel_size=3, padding='same', dilation_rate=2, activation='relu')(conv1)
    conv3 = Conv1D(filters=dim, kernel_size=3, padding='same', dilation_rate=4, activation='relu')(conv2)
    conv4 = Conv1D(filters=dim, kernel_size=3, padding='same', dilation_rate=8, activation='relu')(conv3)
    conv5 = Conv1D(filters=dim, kernel_size=3, padding='same', dilation_rate=16, activation='relu')(conv4)

    pool = GlobalMaxPool1D()(conv5)

    predictions = Dense(5, activation='softmax')(pool)

    model = Model(inputs=input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print("Start training...")
    model.fit(train_x, train_y, batch_size=128, epochs=20, verbose=1, validation_data=[test_x, test_y])
    #model.save_weights('trained_model/yelp/CCG_w.h5')
    score = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy:" + str(score))


# Imdb模型
def model_imdb(model_path, train_path, test_path, max_len, kw_max_len, dim):
    data_dict = preprocess_imdb_clf_data(model_path, train_path, test_path, max_len, kw_max_len, dim)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']

    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    input = Input(shape=(max_len,), dtype='int32')  # 原始文本
    embed = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input)
    embed_dp = Dropout(0.3)(embed)

    conv = Conv1D(filters=300, kernel_size=3, padding='same', activation='relu', strides=1)(embed_dp)
    conv_lstm = Bidirectional(LSTM(max_len,
                                   dropout=0.3,
                                   return_sequences=True,
                                   return_state=False))(conv)

    lstm = Bidirectional(LSTM(max_len,
                              dropout=0.3,
                              return_sequences=True,
                              return_state=False))(embed_dp)

    att = Multiply()([lstm, conv_lstm])
    pool = GlobalMaxPool1D()(att)

    dense = Dense(2 * max_len, activation='relu')(pool)
    dp = Dropout(0.32)(dense)
    predictions = Dense(8, activation='softmax')(dp)

    model = Model(inputs=input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print("Start training...")
    model.fit(train_x, train_y, batch_size=128, epochs=10, verbose=1, validation_data=[test_x, test_y])
    model.save_weights('trained_model/imdb/ours_w.h5')
    score = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy:" + str(score))


# Amazon食品评论模型
def model_amazon_reviews(model_path, reviews_path, max_len, dim):
    data_dict = preprocess_amazon_clf_data(model_path, reviews_path, max_len, dim)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']
    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    input = Input(shape=(max_len,), dtype='int32')  # 原始文本
    embed = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input)
    embed_dp = Dropout(0.3)(embed)

    conv1 = Conv1D(filters=dim, kernel_size=3, padding='same', activation='relu', strides=1)(embed_dp)
    conv2 = Conv1D(filters=dim, kernel_size=3, padding='same', dilation_rate=2, activation='relu')(conv1)
    relu2 = Activation('relu')(conv2)

    conv3 = Conv1D(filters=dim, kernel_size=3, padding='same', dilation_rate=4, activation='relu')(relu2)
    relu3 = Activation('relu')(conv3)

    conv4 = Conv1D(filters=dim, kernel_size=3, padding='same', dilation_rate=8, activation='relu')(relu3)
    relu4 = Activation('relu')(conv4)

    conv5 = Conv1D(filters=dim, kernel_size=3, padding='same', dilation_rate=16, activation='relu')(relu4)
    relu5 = Activation('relu')(conv5)

    pool = GlobalMaxPool1D()(relu5)

    predictions = Dense(5, activation='softmax')(pool)
    model = Model(inputs=input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    model.fit(train_x, train_y, batch_size=64, epochs=10, verbose=1, validation_data=[test_x, test_y])
    #model.save_weights('trained_model/amazon_food/CCG_w.h5')
    score = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy:" + str(score))


# Amazon电子设备评论模型
def model_mobile_reviews(model_path, review_paths, max_len, dim):
    data_dict = preprocess_amazon_mobile_data(model_path, review_paths, max_len, dim)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']

    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    input = Input(shape=(max_len,), dtype='int32')  # 原始文本
    embed = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input)
    embed_dp = Dropout(0.3)(embed)

    lstm = Bidirectional(LSTM(max_len, dropout=0.3, return_sequences=True, return_state=False))(embed_dp)

    conv1 = Conv1D(filters=300, kernel_size=4, padding='same', activation='relu', strides=4)(lstm)
    conv2 = Conv1D(filters=300, kernel_size=3, padding='same', dilation_rate=2, activation='relu')(conv1)
    conv3 = Conv1D(filters=300, kernel_size=3, padding='same', dilation_rate=4, activation='relu')(conv2)
    conv4 = Conv1D(filters=300, kernel_size=3, padding='same', dilation_rate=8, activation='relu')(conv3)

    flat = Flatten()(conv4)

    predictions = Dense(5, activation='softmax')(flat)

    model = Model(inputs=input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print("Start training...")
    model.fit(train_x, train_y, batch_size=32, epochs=40, verbose=1, validation_data=[test_x, test_y])
    #model.save_weights('trained_model/amazon_mobile/CCG_w.h5')
    score = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy:" + str(score))


# DBPedia主题分类
def model_dbpedia(model_path, review_paths, max_len, dim):
    data_dict = preprocess_dbpedia_data(model_path, review_paths, max_len, dim)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']

    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    input = Input(shape=(max_len,), dtype='int32')  # 原始文本
    embed = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input)
    embed_dp = Dropout(0.3)(embed)

    lstm = Bidirectional(LSTM(max_len, dropout=0.3))(embed_dp)

    dense = Dense(2 * max_len, activation='relu')(lstm)
    dp = Dropout(0.32)(dense)
    predictions = Dense(14, activation='softmax')(dp)

    model = Model(inputs=input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print("Start training...")
    model.fit(train_x, train_y, batch_size=128, epochs=10, verbose=1, validation_data=[test_x, test_y])
    model.save_weights('trained_model/dbpedia/lstm_w.h5')
    score = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy:" + str(score))


def model_yahoo(model_path, review_paths, max_len, dim):
    data_dict = preprocess_dbpedia_data(model_path, review_paths, max_len, dim)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']

    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    input = Input(shape=(max_len,), dtype='int32')  # 原始文本
    embed = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input)
    embed_dp = Dropout(0.3)(embed)

    lstm = Bidirectional(LSTM(max_len, dropout=0.3))(embed_dp)

    dense = Dense(2 * max_len, activation='relu')(lstm)
    dp = Dropout(0.32)(dense)
    predictions = Dense(14, activation='softmax')(dp)

    model = Model(inputs=input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print("Start training...")
    model.fit(train_x, train_y, batch_size=128, epochs=10, verbose=1, validation_data=[test_x, test_y])
    model.save_weights('trained_model/yahoo/lstm_w.h5')
    score = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy:" + str(score))


# 返回LSTM与自注意力结合的结果
def model_lstm_with_self_att(embed_dp, max_len):
    hidden_states = embed_dp
    hidden_states = Bidirectional(LSTM(max_len,
                                           dropout=0.3,
                                           return_sequences=True,
                                           return_state=False))(hidden_states)

    # Attention mechanism
    attention = Conv1D(filters=max_len, kernel_size=1, activation='tanh', padding='same', use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros', name="attention_layer1")(hidden_states)
    attention = Conv1D(filters=max_len, kernel_size=1, activation='linear', padding='same',use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros',
                       name="attention_layer2")(attention)
    attention = Lambda(lambda x: softmax(x, axis=1), name="attention_vector")(attention)

    # Apply attention weights
    weighted_sequence_embedding = Dot(axes=[1, 1], normalize=False, name="weighted_sequence_embedding")(
        [attention, hidden_states])

    # Add and normalize to obtain final sequence embedding
    sequence_embedding = Lambda(lambda x: K.l2_normalize(K.sum(x, axis=1)))(weighted_sequence_embedding)

    return sequence_embedding


# 测试Transformer
def test_transformer(model_path, review_paths, max_len, kw_max_len, dim):
    data_dict = preprocess_amazon_mobile_data(model_path, review_paths, max_len, kw_max_len, dim, lda=False)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']

    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    # 提取LDA主题词
    #train_kw_x = data_dict['train_kw_x']
    #test_kw_x = data_dict['test_kw_x']

    print("Building model...")
    transformer = MyTransformerEncoder(vocab_size, max_len, kw_max_len, d_model=dim, word_embed_matrix=embedding_matrix)
    transformer.compile()
    print(transformer.model.summary())
    transformer.fit_model(train_x, train_y, )
    transformer.evaluate(test_x, test_y, )



# 预训练好的词向量模型
glove_300d_path = "yelp_glove/glove.840B.300d.txt"
glove_100d_path = "yelp_glove/glove.6B.100d.txt"
ic_self_trained_model_path = "output_result/up_down_stream.vector"
fast_text_wiki_path = "fast_text_vec/wiki-news-300d-1M.vec"
fast_text_crawl_path = "fast_text_vec/crawl-300d-2M.vec"


# 停用词
stopwords_path = "yelp_glove/stop_words"
stopwords_path_1 = "yelp_glove/stop_words_1"


# 产业链数据
ic_path = "training_data/seg_sent_with_label2"
ic_paths = []
ic_paths.append(ic_path)

# IMDB影评训练、测试数据
imdb_train_path = 'Imdb_reviews/train_data'
imdb_test_path = 'Imdb_reviews/test_data'

# Amazon食品评论数据
amazon_reviews_path = 'Amazon_reviews/food_reviews.csv'
food_reviews = []
food_reviews.append('Amazon_reviews/train.txt')
food_reviews.append('Amazon_reviews/test.txt')

# Amazon电子设备评论数据
mobile_reviews = []
mobile_reviews.append('Amazon_mobile_reviews/train')
mobile_reviews.append('Amazon_mobile_reviews/test')
mobile_reviews.append('Amazon_mobile_reviews/dev')


# 文本蕴含数据
etm_path = "entailment_data/gov_data.txt"

# Yelp评论数据
yelp_paths = []
yelp_paths.append('yelp_new_reviews/train_double')
yelp_paths.append('yelp_new_reviews/test_double')

# DBPedia
dbpedia_paths = ['DBPedia/train', 'DBPedia/test']

# Yahoo
yahoo_paths = ['Yahoo/train', 'Yahoo/test']

if __name__ == '__main__':

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = tf.Session(config=config)
    #KTF.set_session(session)

    #model_yelp_reviews(glove_300d_path, yelp_paths, max_len=120, dim=300)
    #model_imdb(glove_300d_path, imdb_train_path, imdb_test_path, max_len=230, kw_max_len=44, dim=300)
    #model_amazon_reviews(glove_300d_path, food_reviews, max_len=100, dim=300)
    model_mobile_reviews(glove_300d_path, mobile_reviews, max_len=80, dim=300)
    #model_dbpedia(glove_300d_path, dbpedia_paths, max_len=60, dim=300)



