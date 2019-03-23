# -*- encoding: utf-8 -*-

import random
from textgenrnn import textgenrnn


pos_emotion_synonyms = ['perfect', 'good', 'perfectly', 'happy', 'excellent', 'better', 'well', 'nice', 'best', 'awesome',
                        'great', 'friendly', 'fine']
neg_emotion_synonyms = ['bad', 'awful', 'broken', 'issue', 'problem', 'locked', 'poor', 'disappointed', 'die', 'dead',
                        'wrong']
verb_synonyms = ['love', 'like', 'thank', 'recommend']
noun_synonyms = ['phone', 'charge', 'iphone', 'usb', 'charging', 'screen', 'cable', 'charger', 'device', 'power',
                 'battery', 'app', 'sim', 'button', 'android', 'network', 'camera']


def augment_training_set(train_x, train_y):
    aug_train_x = []
    aug_train_y = []
    for i in range(0, len(train_x)):
        x = train_x[i]
        y = train_y[i]
        # 挑出包含在原句中的词语
        contained_pos_emt_words = []
        contained_neg_emt_words = []
        contained_vb_words = []
        contained_n_words = []

        for emt_word in pos_emotion_synonyms:
            if ' '+emt_word+' ' in x:
                contained_pos_emt_words.append(emt_word)
        for emt_word in neg_emotion_synonyms:
            if ' '+emt_word+' ' in x:
                contained_neg_emt_words.append(emt_word)
        for vb_word in verb_synonyms:
            if ' '+vb_word+' ' in x:
                contained_vb_words.append(vb_word)

        for n_word in noun_synonyms:
            if ' '+n_word+' ' in x:
                contained_n_words.append(n_word)

        # 筛出不包含在原句中的词语
        random_pos_emt_words = []
        random_neg_emt_words = []
        random_vb_words = []
        random_n_words = []

        for emt_word in pos_emotion_synonyms:
            if emt_word not in contained_pos_emt_words:
                random_pos_emt_words.append(emt_word)
        for emt_word in neg_emotion_synonyms:
            if emt_word not in contained_neg_emt_words:
                random_neg_emt_words.append(emt_word)
        for vb_word in verb_synonyms:
            if vb_word not in contained_vb_words:
                random_vb_words.append(vb_word)

        for n_word in noun_synonyms:
            if n_word not in contained_n_words:
                random_n_words.append(n_word)

        aug_train_x.append(x)
        aug_train_y.append(y)


        # 通过替换生成新样本扩充训练集
        if len(random_pos_emt_words) > 0:
            for emt_word in contained_pos_emt_words:
                index = random.randint(0, len(random_pos_emt_words)-1)
                rp_emt_word = random_pos_emt_words[index]
                new_x = x.replace(emt_word, rp_emt_word)
                aug_train_x.append(new_x)
                aug_train_y.append(y)

        if len(random_neg_emt_words) > 0:
            for emt_word in contained_neg_emt_words:
                index = random.randint(0, len(random_neg_emt_words)-1)
                rp_emt_word = random_neg_emt_words[index]
                new_x = x.replace(emt_word, rp_emt_word)
                aug_train_x.append(new_x)
                aug_train_y.append(y)

        if len(random_vb_words) > 0:
            for vb_word in contained_vb_words:
                index = random.randint(0, len(random_vb_words)-1)
                rp_vb_word = random_vb_words[index]
                new_x = x.replace(vb_word, rp_vb_word)
                aug_train_x.append(new_x)
                aug_train_y.append(y)

        if len(random_n_words) > 0:
            for n_word in contained_n_words:
                index = random.randint(0, len(random_n_words)-1)
                rp_n_word = random_n_words[index]
                new_x = x.replace(n_word, rp_n_word)
                aug_train_x.append(new_x)
                aug_train_y.append(y)

        print('Finish augmenting sentence:'+str(i))

    label1_num = 0
    label2_num = 0
    label3_num = 0
    label4_num = 0
    label5_num = 0
    for y in aug_train_y:
        if y == 0:
            label1_num += 1
        elif y == 1:
            label2_num += 1
        elif y == 2:
            label3_num += 1
        elif y == 3:
            label4_num += 1
        elif y == 4:
            label5_num += 1
    return aug_train_x, aug_train_y

