# -*- encoding:utf-8 -*-


from gensim import corpora
from gensim.models import LdaModel
from util import rm_useless_tokens
import re
import numpy as np
from util import get_contained_keywords


# 对所有类别训练LDA模型，返回所有主题词的list
def model_lda_to_words(cat1_rvs, cat2_rvs, cat3_rvs, cat4_rvs, cat5_rvs):
    # 读取停用词
    stopwords = []
    stopwords_path = "yelp_glove/stop_words_1"
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.append(line.replace('\n', ''))
    print("Finish loading stopwords.")

    # LDA建模
    lda1 = model_lda_for_single_cat(cat1_rvs, stopwords, 1)
    print("Finish modeling cat1 with LDA.")
    lda2 = model_lda_for_single_cat(cat2_rvs, stopwords, 1)
    print("Finish modeling cat2 with LDA.")
    lda3 = model_lda_for_single_cat(cat3_rvs, stopwords, 1)
    print("Finish modeling cat3 with LDA.")
    lda4 = model_lda_for_single_cat(cat4_rvs, stopwords, 1)
    print("Finish modeling cat4 with LDA.")
    lda5 = model_lda_for_single_cat(cat5_rvs, stopwords, 1)
    print("Finish modeling cat5 with LDA.")

    # 获得每个类别中LDA主题中的词语
    dict1 = get_lda_topic_words(lda1, 1, 500)
    dict2 = get_lda_topic_words(lda2, 1, 500)
    dict3 = get_lda_topic_words(lda3, 1, 500)
    dict4 = get_lda_topic_words(lda4, 1, 500)
    dict5 = get_lda_topic_words(lda5, 1, 500)

    '''
    # 将所有lda类别的字典加入一个字典列表
    dict_list = []
    for dict in dict1:
        dict_list.append(dict)
    for dict in dict2:
        dict_list.append(dict)
    for dict in dict3:
        dict_list.append(dict)
    for dict in dict4:
        dict_list.append(dict)
    for dict in dict5:
        dict_list.append(dict)

    '''
    # 将所有类别LDA主题中词语求并集存于keywords
    keywords = []
    for dict in dict1:
        for word in dict:
            keywords.append(word)
    print("Finish union cat1,")
    for dict in dict2:
        for word in dict:
            if word not in keywords:
                keywords.append(word)
    print("Finish union cat2,")
    for dict in dict3:
        for word in dict:
            if word not in keywords:
                keywords.append(word)
    print("Finish union cat3,")
    for dict in dict4:
        for word in dict:
            if word not in keywords:
                keywords.append(word)
    print("Finish union cat4,")
    for dict in dict5:
        for word in dict:
            if word not in keywords:
                keywords.append(word)
    print("Finish union cat5,")

    return keywords


# 对所有类别训练LDA模型，返回主题dict的list
def model_lda_to_dicts(cat1_rvs, cat2_rvs, cat3_rvs, cat4_rvs, cat5_rvs):
    # 读取停用词
    stopwords = []
    stopwords_path = "yelp_glove/stop_words_1"
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.append(line.replace('\n', ''))
    print("Finish loading stopwords.")

    # LDA建模
    lda1 = model_lda_for_single_cat(cat1_rvs, stopwords, 5)
    print("Finish modeling cat1 with LDA.")
    lda2 = model_lda_for_single_cat(cat2_rvs, stopwords, 5)
    print("Finish modeling cat2 with LDA.")
    lda3 = model_lda_for_single_cat(cat3_rvs, stopwords, 5)
    print("Finish modeling cat3 with LDA.")
    lda4 = model_lda_for_single_cat(cat4_rvs, stopwords, 5)
    print("Finish modeling cat4 with LDA.")
    lda5 = model_lda_for_single_cat(cat5_rvs, stopwords, 5)
    print("Finish modeling cat5 with LDA.")

    # 获得每个类别中LDA主题中的词语
    dict1 = get_lda_topic_words(lda1, 5, 50)
    dict2 = get_lda_topic_words(lda2, 5, 50)
    dict3 = get_lda_topic_words(lda3, 5, 50)
    dict4 = get_lda_topic_words(lda4, 5, 50)
    dict5 = get_lda_topic_words(lda5, 5, 50)

    # 将所有lda类别的字典加入一个字典列表
    dict_list = []
    for dict in dict1:
        dict_list.append(dict)
    for dict in dict2:
        dict_list.append(dict)
    for dict in dict3:
        dict_list.append(dict)
    for dict in dict4:
        dict_list.append(dict)
    for dict in dict5:
        dict_list.append(dict)

    return dict_list


# 改变思路 #################################################
# 对所有训练集训练LDA模型
def model_lda_for_all(train_x):
    # 读取停用词
    stopwords = []
    stopwords_path = "yelp_glove/stop_words_1"
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.append(line.replace('\n', ''))
    print("Finish loading stopwords.")

    # LDA建模
    print('Training LDA...')
    lda1 = model_lda_for_single_cat(train_x, stopwords, 50)
    print("Finish LDA modeling.")

    # 获得每个类别中LDA主题中的词语
    dict_list = get_lda_topic_words(lda1, 50, 50)

    return dict_list


# 找到与句子最相似的LDA类
def get_similar_lda_cat(ori_x, lda_dict_list):
    similar_lda_words = []
    ct_words_list = []
    for i in range(0, len(ori_x)):
        review = ori_x[i]
        lda_words = get_similar_lda_for_single_sent(review, lda_dict_list)
        if len(lda_words) != 0:
            contained_keywords = get_contained_keywords(review, lda_words,)  # 获取包含在LDA中的句子词语
            similar_lda_words.append(lda_words)
            ct_words_list.append(contained_keywords)
        else:
            similar_lda_words.append('')
            ct_words_list.append('')

    return similar_lda_words, ct_words_list


# 对一个句子找到最相似的LDA类别
def get_similar_lda_for_single_sent(review, lda_dict_list):
    words = review.split()
    contained_word_nums = []  # 记录每个LDA类包含该句子词语的数量
    contained_word_probs = []  # 记录每个LDA类包含该句子词语概率总和
    for lda_dict in lda_dict_list:
        contained_word_num = 0
        contained_word_prob = 0
        for word in words:
            if word in lda_dict.keys():
                contained_word_num += 1
                contained_word_prob += float(lda_dict[word])
        contained_word_nums.append(contained_word_num)
        contained_word_probs.append(contained_word_prob)

    # 找出包含最多词语的LDA类
    most_word_index = 0
    most_words_num = 0
    most_words_prob = 0
    for i in range(0, len(contained_word_nums)):
        num = contained_word_nums[i]
        prob = contained_word_probs[i]
        if num >= most_words_num and prob > most_words_prob:
            most_words_num = num
            most_words_prob = prob
            most_word_index = i

    return lda_dict_list[most_word_index].keys()


# 对所有句子匹配LDA类别，并生成样本
def lda_match(ori_x, lda_dict, embed_dict):
    for sent in ori_x:
        words = sent.split()
        word_lda_dict = {}  # key为句子中的词语，value为对应的LDA类别
        f_word_lda_dict = {}  # 经过过滤后的LDA结果，key为句子中的词语，value为对应的LDA词语
        # 对每个词语计算其匹配的LDA类别
        for word in words:
            word_lda = match_word_lda(word, lda_dict)
            word_lda_dict[word] = word_lda
        sent_lda_words = word_lda_dict.keys()  # 该句中有对应LDA类的词语
        for word in word_lda_dict.keys():
            lda_words = word_lda_dict[word]  # 初始LDA类中词语
            f_lda_words = filter_word_lda(sent_lda_words, lda_words, embed_dict)  # 过滤后LDA的词语
            f_word_lda_dict[word] = f_lda_words
        print()


# 对一个词语匹配LDA类别
def match_word_lda(word, lda_dict):
    max_prob = 0.0  # 标记该词在各LDA类中的概率
    word_lda = None  # 最匹配该词的LDA类别
    for dict in lda_dict:
        if word in dict.keys():
            if max_prob < float(dict[word]):
                max_prob = float(dict[word])
                word_lda = dict.keys()
    return word_lda


# 对每个词语的LDA类进行过滤
def filter_word_lda(sent_lda_words, lda_words, embed_dict):
    # 计算句子包含在LDA中词语的词向量均值
    embed_sum = np.zeros(300)
    i = 0
    for c_word in sent_lda_words:
        try:
            c_embed = embed_dict[c_word]
            embed_sum += c_embed
            i += 1
        except KeyError:
            pass
    avg_embed = embed_sum / i
    # 计算LDA中词向量和上面的词向量均值的余弦相似度和欧式距离
    word_sim_dict = {}
    word_l2_dict = {}
    for word in lda_words:
        try:
            embed = embed_dict[word]
            sim = cosine_similarity(avg_embed, embed)
            l2 = l2_distance(avg_embed, embed)
            word_sim_dict[word] = sim
            word_l2_dict[word] = l2
        except KeyError:
            pass
    # 计算余弦相似度和欧式距离的均值
    avg_sim = sum(word_sim_dict.values()) / len(word_sim_dict)
    avg_l2 = sum(word_l2_dict.values()) / len(word_l2_dict)
    # 过滤
    f_lda_words = []
    for word in word_sim_dict:
        if word_sim_dict[word] >= avg_sim and word_l2_dict[word] <= avg_l2:
            f_lda_words.append(word)
    return f_lda_words


########################################################################
# 对所有类别训练LDA模型
def model_lda_for_imdb_cats(cat1_rvs, cat2_rvs, cat3_rvs, cat4_rvs, cat5_rvs, cat6_rvs, cat7_rvs, cat8_rvs):
    # 读取停用词
    stopwords = []
    stopwords_path = "yelp_glove/stop_words_1"
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.append(line.replace('\n', ''))
    print("Finish loading stopwords.")

    # LDA建模
    lda1 = model_lda_for_single_cat(cat1_rvs, stopwords, 1)
    print("Finish modeling cat1 with LDA.")
    lda2 = model_lda_for_single_cat(cat2_rvs, stopwords, 1)
    print("Finish modeling cat2 with LDA.")
    lda3 = model_lda_for_single_cat(cat3_rvs, stopwords, 1)
    print("Finish modeling cat3 with LDA.")
    lda4 = model_lda_for_single_cat(cat4_rvs, stopwords, 1)
    print("Finish modeling cat4 with LDA.")
    lda5 = model_lda_for_single_cat(cat5_rvs, stopwords, 1)
    print("Finish modeling cat5 with LDA.")
    lda6 = model_lda_for_single_cat(cat6_rvs, stopwords, 1)
    print("Finish modeling cat6 with LDA.")
    lda7 = model_lda_for_single_cat(cat7_rvs, stopwords, 1)
    print("Finish modeling cat7 with LDA.")
    lda8 = model_lda_for_single_cat(cat8_rvs, stopwords, 1)
    print("Finish modeling cat8 with LDA.")

    # 获得每个类别中LDA主题中的词语
    dict1 = get_lda_topic_words(lda1, 1, 500)
    dict2 = get_lda_topic_words(lda2, 1, 500)
    dict3 = get_lda_topic_words(lda3, 1, 500)
    dict4 = get_lda_topic_words(lda4, 1, 500)
    dict5 = get_lda_topic_words(lda5, 1, 500)
    dict6 = get_lda_topic_words(lda6, 1, 500)
    dict7 = get_lda_topic_words(lda7, 1, 500)
    dict8 = get_lda_topic_words(lda8, 1, 500)

    # 将所有类别LDA主题中词语求并集存于keywords
    keywords = []
    for dict in dict1:
        for word in dict:
            keywords.append(word)
    print("Finish union cat1.")
    for dict in dict2:
        for word in dict:
            if word not in keywords:
                keywords.append(word)
    print("Finish union cat2.")
    for dict in dict3:
        for word in dict:
            if word not in keywords:
                keywords.append(word)
    print("Finish union cat3.")
    for dict in dict4:
        for word in dict:
            if word not in keywords:
                keywords.append(word)
    print("Finish union cat4.")
    for dict in dict5:
        for word in dict:
            if word not in keywords:
                keywords.append(word)
    print("Finish union cat5.")
    for dict in dict6:
        for word in dict:
            if word not in keywords:
                keywords.append(word)
    print("Finish union cat6.")
    for dict in dict7:
        for word in dict:
            if word not in keywords:
                keywords.append(word)
    print("Finish union cat7.")
    for dict in dict8:
        for word in dict:
            if word not in keywords:
                keywords.append(word)
    print("Finish union cat8.")

    return keywords


# 对一个类别训练LDA模型
def model_lda_for_single_cat(reviews, stopwords, topic_num):
    for i in range(0, len(reviews)):
        temp = rm_useless_tokens(reviews[i].lower()) # 替换无用符号
        temp = temp.replace('!', '').replace('?', '')
        temp = re.sub(r'\d+', '', temp)  # 替换数字
        f_review = remove_stop_words(temp, stopwords)
        if f_review != '':
            reviews[i] = f_review

    tokens = []
    for rv in reviews:
        tokens.append(rv.split())

    dic = corpora.Dictionary(tokens)  # 生成文档词典，每一个词与一个索引值对应
    corpus = [dic.doc2bow(t) for t in tokens]  # 词频统计，转换为空间向量格式
    lda = LdaModel(corpus=corpus, id2word=dic, num_topics=topic_num, alpha='auto', passes=1)

    return lda


# 去除停用词（一些与主题分类无关的词）
def remove_stop_words(one_line, stop_words):
    for stop_word in stop_words:
        words = one_line.split()
        if len(words) == 0:  # 该句所有词都被停用词过滤则返回空
            return ''
        if words[0] == stop_word:
            one_line = one_line.replace(stop_word+" ", "")
        elif words[len(words)-1] == stop_word:
            one_line = one_line.replace(' '+stop_word, '')
        else:
            one_line = one_line.replace(" "+stop_word+" ", " ")
        one_line = one_line.replace(' it it ', ' ')

    return one_line


# 获得各个主题的词语和其概率，返回一个字典的列表
def get_lda_topic_words(lda, topic_num, word_num):
    result_list = lda.show_topics(num_topics=topic_num, num_words=word_num)

    dic_list = []
    for tup in result_list:
        dic = {}
        pro_words = tup[1].split("+")
        for pro_word in pro_words:
            word = pro_word.split("*")
            dic[word[1].replace('"', "").replace(" ", "")] = word[0]
        dic_list.append(dic)

    return dic_list


# 对每一个样本获取其相应的lda表示，其中参数dict_list为LDA类别字典的列表
# 每一个元素为一个字典，key为词语，value为该词语在该主题下的概率
def get_fit_lda_cat_words(ori_x, ori_y, dict_list, embed_dict=None):
    train_ori_x = []
    train_kw_x = []
    train_kw_y = []
    contained_w_list = []
    f_contained_w_list = []
    for i in range(0, len(ori_x)):
        review = ori_x[i]
        lda_words = get_fit_lda_cat(review, dict_list)  # 需要进行split操作转换为list
        if lda_words is not None:
            contained_keywords = get_contained_keywords(review, lda_words.split(), )  # 获取包含在LDA中的句子词语
            # 过滤句子词语
            if len(contained_keywords) != 1:
                f_contained_keywords = filter_contained_words(lda_words.split()[0], contained_keywords, embed_dict)
                # 过滤LDA类中词语
                f_lda_words = filter_lda_cat_by_sent(f_contained_keywords, lda_words.split(), embed_dict)
                f_contained_w_list.append(f_contained_keywords)
            else:
                f_lda_words = filter_lda_cat_by_sent(contained_keywords, lda_words.split(), embed_dict)
                f_contained_w_list.append(contained_keywords)
            train_kw_x.append(f_lda_words)
            train_kw_y.append(ori_y[i])
            train_ori_x.append(review)
            contained_w_list.append(contained_keywords)
        else:
            contained_w_list.append(review)
            f_contained_w_list.append(review)
            train_kw_x.append(review)
            train_kw_y.append(ori_y[i])
            train_ori_x.append(review)
        print("Finish loading lda representation:"+str(i))

    return train_ori_x, train_kw_x, train_kw_y, contained_w_list, f_contained_w_list


# 找出与当前样本最匹配的LDA类，返回该LDA中的所有词语
def get_fit_lda_cat(review, dict_list):
    dict_word = {}
    dict_prob = {}
    for dict in dict_list:
        word_num = 0
        prob_sum = 0
        for word in dict:
            if ' '+word+' ' in review:
                word_num += 1
                prob_sum += float(dict[word])
        words = ' '.join(dict.keys())
        dict_word[words] = word_num
        dict_prob[words] = prob_sum

    if len(dict_word) == 0:  # 该句子没有任何一个词语包含在任意一组LDA类别中
        return None

    most_word_dicts = []
    most_word_num = sorted(dict_word.values(), reverse=True)[0]
    for words in dict_word:
        if dict_word[words] == most_word_num:
            most_word_dicts.append(words)

    if len(most_word_dicts) == 1:
        return most_word_dicts[0]

    most_prob = 0
    fit_words = None
    for words in most_word_dicts:
        if dict_prob[words] > most_prob:
            most_prob = dict_prob[words]
            fit_words = words
    return fit_words


# 根据当前句子包含在lda的词语来过滤lda
def filter_lda_cat_by_sent(contained_words, lda_words, embed_dict):
    # 计算句子包含在LDA中词语的词向量均值
    embed_sum = np.zeros(300)
    i = 0
    for c_word in contained_words:
        try:
            c_embed = embed_dict[c_word]
            embed_sum += c_embed
            i += 1
        except KeyError:
            pass
    avg_embed = embed_sum/i
    # 计算LDA中词向量和上面的词向量均值的余弦相似度和欧式距离
    word_sim_dict = {}
    word_l2_dict = {}
    for word in lda_words:
        try:
            embed = embed_dict[word]
            sim = cosine_similarity(avg_embed, embed)
            l2 = l2_distance(avg_embed, embed)
            word_sim_dict[word] = sim
            word_l2_dict[word] = l2
        except KeyError:
            pass
    # 计算余弦相似度和欧式距离的均值
    avg_sim = sum(word_sim_dict.values())/len(word_sim_dict)
    avg_l2 = sum(word_l2_dict.values())/len(word_l2_dict)
    # 过滤
    f_lda_words = []
    for word in word_sim_dict:
        if word_sim_dict[word] >= avg_sim and word_l2_dict[word] <= avg_l2:
            f_lda_words.append(word)
    return f_lda_words


# 找出包含在句子中的LDA词语，根据该LDA类的中心词再来过滤包含的词语
def filter_contained_words(c_word, contained_words, embed_dict):
    word_sim_dict = {}
    word_l2_dict = {}
    for word in contained_words:
        if word == c_word:
            continue
        try:  # 计算每个包含词与中心词的余弦相似度和欧式距离
            c_w_embed = embed_dict[c_word]
            w_embed = embed_dict[word]
            sim = cosine_similarity(c_w_embed, w_embed)
            l2 = l2_distance(c_w_embed, w_embed)
            word_sim_dict[word] = sim
            word_l2_dict[word] = l2
        except KeyError:
            pass
    # 没找到上述词的词向量则直接返回传入的参数
    if len(word_sim_dict) == 0 and len(word_l2_dict) == 0:
        return contained_words
    avg_sim = sum(word_sim_dict.values())/len(word_sim_dict)  # 计算均值
    avg_l2 = sum(word_l2_dict.values())/len(word_l2_dict)
    # 挑选
    f_contained_words = []
    if c_word in contained_words:
        f_contained_words.append(c_word)
    for word in contained_words:
        if word == c_word:
            continue
        try:
            cur_sim = word_sim_dict[word]
            cur_l2 = word_l2_dict[word]
            if cur_sim >= avg_sim and cur_l2 <= avg_l2:
                f_contained_words.append(word)
        except KeyError:
            pass
    return f_contained_words


# 统计在一个标签的所有LDA类别里共同包含的词语
def get_same_words(dict_list):
    all_words_dict = {}
    for dict in dict_list:
        words = dict.keys()
        for word in words:
            if word not in all_words_dict.keys():
                all_words_dict[word] = 0

    for dict in dict_list:
        words = dict.keys()
        for word in words:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1

    return all_words_dict


# 对每一个LDA类别根据中心词进行过滤，去除一些无关的词语
def filter_lda_cats_by_center_word(dict_list, word_embed_dict):
    print("Begin to filtering lda words...")
    ft_lda_dict_list = []
    for dict in dict_list:
        # 找出该lda类别的中心词
        most_prob = sorted(dict.values(), reverse=True)[0]
        c_word = None
        for word in dict.keys():
            if dict[word] == most_prob:
                c_word = word
                break
        dict_words = dict.keys()
        # 计算该lda类别中其他词与中心词的余弦相似度和欧式距离
        word_sim_dict = {}
        word_l2_dict = {}
        for word in dict_words:
            if word != c_word:
                try:
                    word_embed = word_embed_dict[word]
                    c_word_embed = word_embed_dict[c_word]
                    cos_sim = cosine_similarity(word_embed, c_word_embed)
                    l2_dis = l2_distance(word_embed, c_word_embed)
                    word_sim_dict[word] = cos_sim
                    word_l2_dict[word] = l2_dis
                except Exception:
                    pass
        # 在LDA类别中挑选出与中心词相似度大于平均相似度的词语
        sim_word_prob_dict = {}
        sims_sum = sum(word_sim_dict.values())  # 计算平均余弦相似度
        avg_sim = sims_sum/len(word_sim_dict)
        l2_sum = sum(word_l2_dict.values())  # 计算平均欧式距离
        avg_l2 = l2_sum/len(word_l2_dict)
        for word in word_sim_dict.keys():
            if word_sim_dict[word] >= avg_sim and word_l2_dict[word] <= avg_l2:
                sim_word_prob_dict[word] = dict[word]
        ft_lda_dict_list.append(sim_word_prob_dict)
    return ft_lda_dict_list


# 对每一个LDA类别根据该LDA类别词向量均值进行过滤
def filter_lda_cats_by_mean_word_embed(dict_list, word_embed_dict):
    print("Begin to filtering lda words...")
    ft_lda_dict_list = []
    for dict in dict_list:
        # 计算一个LDA类别中所有词向量的均值
        word_embed_sum = np.zeros(300)
        for word in dict.keys():
            try:
                w_embed = word_embed_dict[word]
                word_embed_sum += w_embed
            except KeyError:
                pass
        word_embed_avg = word_embed_sum/len(dict)
        # 计算LDA类别中所有词向量与均值之间的余弦相似度
        word_sim_dict = {}
        word_l2_dict = {}
        for word in dict.keys():
            try:
                w_embed = word_embed_dict[word]
                sim = cosine_similarity(w_embed, word_embed_avg)
                l2 = l2_distance(w_embed, word_embed_avg)
                word_sim_dict[word] = sim
                word_l2_dict[word] = l2
            except KeyError:
                pass
        # 计算平均余弦相似度和平均欧式距离
        avg_sim = sum(word_sim_dict.values())/ len(word_sim_dict)
        avg_l2 = sum(word_l2_dict.values()) / len(word_l2_dict)
        # 挑选LDA中符合条件的词语
        word_prob_dict = {}
        for word in word_sim_dict.keys():
            if word_sim_dict[word] >= avg_sim and word_l2_dict[word] <= avg_l2:
                word_prob_dict[word] = dict[word]
        ft_lda_dict_list.append(word_prob_dict)
    return ft_lda_dict_list


# 计算余弦相似度
def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)


# 计算欧式距离
def l2_distance(vector1, vector2):
    sum = 0
    for a, b in zip(vector1, vector2):
        sum += (a-b) ** 2
    return sum ** 0.5
