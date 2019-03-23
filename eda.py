# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
from random import shuffle
random.seed(1)

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

# for the first time you use wordnet
'''
import nltk
print('Downloading wordnet...')
nltk.download('wordnet')
'''
from nltk.corpus import wordnet


def synonym_replacement(words, n):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in stop_words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			#print("replaced", random_word, "with", synonym)
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break

	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')

	return new_words


# 替换LDA关键词
def keywords_synonym_replace(words, keywords, n):
	new_words = words.copy()
	num_replaced = 0
	if 0 < len(keywords) < n:  # 若关键词数量小于待替换词的数量
		# 先替换关键词
		for keyword in keywords:
			synonyms = get_synonyms(keyword)
			if len(synonyms) > 1:
				synonym = random.choice(list(synonyms))
				new_words = [synonym if word == keyword else word for word in new_words]
				num_replaced += 1
		# 替换关键词以外的词
		random_word_list = list(set([word for word in words if word not in stop_words and word not in keywords]))
		random.shuffle(random_word_list)
		for random_word in random_word_list:
			synonyms = get_synonyms(random_word)
			if len(synonyms) >= 1:
				synonym = random.choice(list(synonyms))
				new_words = [synonym if word == random_word else word for word in new_words]
				num_replaced += 1
			if num_replaced >= n:
				break
	else:  # 关键词数量大于待替换词的数量
		for keyword in keywords:
			synonyms = get_synonyms(keyword)
			if len(synonyms) > 1:
				synonym = random.choice(list(synonyms))
				new_words = [synonym if word == keyword else word for word in new_words]
				num_replaced += 1
			if num_replaced >= n:
				break

	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')

	return new_words


def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word):
		for l in syn.lemmas():
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym)
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################


def random_deletion(words, p, keywords=None):
	# obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	# randomly delete words with probability p
	new_words = []
	for word in words:
		if keywords is None:
			r = random.uniform(0, 1)
			if r > p:
				new_words.append(word)
		else:
			if word not in keywords:
				r = random.uniform(0, 1)
				if r > p:
					new_words.append(word)

	# if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################


def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words


def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################


def random_insertion(words, n, keywords=None):
	new_words = words.copy()
	for _ in range(n):  # 插入n次
		if keywords is None:
			add_word(new_words)
		else:
			add_keyword_syn(new_words, keywords)
	return new_words


def add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		random_word = new_words[random.randint(0, len(new_words)-1)]
		synonyms = get_synonyms(random_word)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)


# 插入keyword的同义词
def add_keyword_syn(new_words, keywords):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		keyword = new_words[random.randint(0, len(keywords) - 1)]
		synonyms = get_synonyms(keyword)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words) - 1)
	new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################


def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9, keywords=None):
	
	sentence = get_only_chars(sentence)
	words = sentence.split(' ')
	words = [word for word in words if word is not '']
	num_words = len(words)
	
	augmented_sentences = []
	num_new_per_technique = int(num_aug/4)+1
	n_sr = max(1, int(alpha_sr*num_words))
	n_ri = max(1, int(alpha_ri*num_words))
	n_rs = max(1, int(alpha_rs*num_words))

	# sr
	for _ in range(num_new_per_technique):
		if keywords is None:
			a_words = synonym_replacement(words, n_sr)
		else:
			a_words = keywords_synonym_replace(words, keywords, n_sr)
		augmented_sentences.append(' '.join(a_words))

	# ri
	for _ in range(num_new_per_technique):
		if keywords is None:
			a_words = random_insertion(words, n_ri)
		else:
			a_words = random_insertion(words, n_ri, keywords)
		augmented_sentences.append(' '.join(a_words))

	# rs
	for _ in range(num_new_per_technique):
		a_words = random_swap(words, n_rs)
		augmented_sentences.append(' '.join(a_words))

	# rd
	for _ in range(num_new_per_technique):
		a_words = random_deletion(words, p_rd, keywords)
		augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
	shuffle(augmented_sentences)

	# trim so that we have the desired number of augmented sentences
	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	# append the original sentence
	augmented_sentences.append(sentence)

	return augmented_sentences