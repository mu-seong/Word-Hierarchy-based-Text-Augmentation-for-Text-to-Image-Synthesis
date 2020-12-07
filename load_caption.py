# caption augmentation for text to image generation
# reference to Jason Wei and Kai Zou
# museong kim

import random
from random import shuffle
random.seed(1)
#for the first time you use wordnet
#import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.tag import pos_tag


#stop word
stop_words = set(stopwords.words('english'))

#tokenizer
tokenizer = RegexpTokenizer("[\w']+")


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
        if char in 'qwertyuiopasdfghjklzxcvbnm':
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

def synonym_replacement(words, n, sentences):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in stop_words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word, sentences)
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

def get_synonyms(word, sentences):
	synonyms = set()
	tag = pos_tag(tokenizer.tokenize(sentences))

	noun_list = []
	etc_list = []

	for t in tag:
		if t[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$']:
			noun_list.append(t[0])
		else:
			etc_list.append(t[0])

	if word in noun_list:
		i = noun_list.index(word)
		noun_word = noun_list[i]

		for syn in wordnet.synsets(noun_word):
			for l in syn.hypernyms(): 
				synonym = l.name().replace("_", " ").replace("-", " ").lower()
				synonym = "".join([char for char in synonym if char in 'qwertyuiopasdfghjklzxcvbnm'])
				synonym = synonym[:-1]
				synonyms.add(synonym)
	else:
		i = etc_list.index(word)
		etc_word = etc_list[i]
		for syn in wordnet.synsets(etc_word):
			for l in syn.lemmas(): 
				synonym = l.name().replace("_", " ").replace("-", " ").lower()
				#synonym = "".join([char for char in synonym if char in 'qwertyuiopasdfghjklzxcvbnm'])
				synonyms.add(synonym)
				
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)
	
########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr, num_aug):	

	sentences = get_only_chars(sentence)
	words = tokenizer.tokenize(sentences)
	#words = sentences.split(' ')
	#words = [word for word in words if word is not '']
	num_words = len(words)
	
	augmented_sentences = []
	num_new_per_technique = int(num_aug/4)+1
	n_sr = max(1, int(alpha_sr*num_words))

	#sr
	for _ in range(num_new_per_technique):
		a_words = synonym_replacement(words, n_sr, sentences)
		augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [get_only_chars(sentences) for sentences in augmented_sentences]
	shuffle(augmented_sentences)

	#trim so that we have the desired number of augmented sentences
	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	#append the original sentence
	augmented_sentences.append(sentences)

	return augmented_sentences