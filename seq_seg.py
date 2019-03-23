# -*- encoding: utf-8 -*-


def seg_sequences(word_embed_dict, sentences):
    i = 0
    new_sequences = []
    for sent in sentences:
        dict, avg_sim = seg_single_sent(word_embed_dict, sent)
        if dict is None:
            new_sequences.append(sent)
            continue
        phrases = print_sim_phrase(dict, avg_sim)
        new_sent = ''
        for p in phrases:
            phrase = ' '.join(p)
            new_sent += ' '+phrase
        new_sequences.append(new_sent)
        print('Finished:'+str(i))
        i += 1
        if i == 5000:
            break
    print()
    return new_sequences


# 分割单个句子
def seg_single_sent(word_embed_dict, sent):
    words = sent.split()
    sim_sum = 0
    bi_gram_sim_dict = {}
    for i in range(1, len(words)):
        cur_word = words[i]
        pre_word = words[i-1]
        if cur_word in word_embed_dict and pre_word in word_embed_dict:
            cur_word_embed = word_embed_dict[cur_word]
            pre_word_embed = word_embed_dict[pre_word]
            sim = cosine_similarity(cur_word_embed, pre_word_embed)
            sim_sum += sim
            bi_gram_sim_dict[(pre_word, cur_word)] = sim
    if len(bi_gram_sim_dict) == 0:
        return None, None
    avg_sim = sim_sum/len(bi_gram_sim_dict)
    return bi_gram_sim_dict, avg_sim


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


def print_sim_phrase(dict, avg_sim):
    phrases = []
    cur_phrase = []
    for phrase in dict:
        word1 = phrase[0]
        word2 = phrase[1]
        if dict[phrase] > avg_sim:
            # print(word1 + " " + word2 + ' ' +str(dict[phrase]))
            if len(cur_phrase) == 0:
                cur_phrase.append(word1)
                cur_phrase.append(word2)
            else:
                if word1 == cur_phrase[len(cur_phrase)-1]:  # 该词组的第一个词与最后一个词是否相同
                    cur_phrase.append(word2)
                    if len(cur_phrase) == 4:  # 当前词组长度为4
                        phrases.append(cur_phrase.copy())
                        cur_phrase = []
                else:
                    phrases.append(cur_phrase.copy())
                    cur_phrase = []
                    cur_phrase.append(word1)
                    cur_phrase.append(word2)
        else:
            if len(cur_phrase) != 0:
                phrases.append(cur_phrase.copy())
                cur_phrase = []
                phrases.append([word2])
            else:
                phrases.append([word1])
                phrases.append([word2])

    new_phrases = []  # 去除重复的词语
    for i in range(0, len(phrases)-1):
        cur = phrases[i]
        nt = phrases[i+1]
        if cur[0] != nt[0] and cur[len(cur)-1] != nt[0] and cur not in new_phrases:
            new_phrases.append(cur)
            new_phrases.append(nt)
        elif cur[0] == nt[0] and len(cur) < len(nt) and cur in new_phrases:
            new_phrases.remove(cur)
            new_phrases.append(nt)
        elif cur in new_phrases and nt not in new_phrases:
            new_phrases.append(nt)
        elif len(nt) == 1 and len(cur) != 1 and cur[len(cur)-1] == nt[0]:
            new_phrases.append(cur)

    extras = []  # 标记出经过一次过滤后存在的重复短语下标
    for i in range(0, len(new_phrases)-1):
        cur = new_phrases[i]
        nt = new_phrases[i+1]
        if len(nt) == 1 and cur[len(cur)-1] == nt[0]:
            extras.append(i+1)

    new2_phrases = []  # 保存最终过滤结果
    for i in range(0, len(new_phrases)):
        if i not in extras:
            add_words_num = 4 - len(new_phrases[i])
            for j in range(add_words_num):  # 添加占位符
                new_phrases[i].append('ph')
            new2_phrases.append(new_phrases[i])

    return new2_phrases
