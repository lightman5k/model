import nltk
import numpy as np
import pickle as pkl
import itertools
from collections import Counter, OrderedDict
from keras.preprocessing.sequence import pad_sequences
import re
import gensim
from nltk.corpus import stopwords
import string


punctuation_list = list(string.punctuation)
# English stop words lists
stop_words = stopwords.words('english')
for punctuation in punctuation_list:
    stop_words.append(punctuation)
    

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def sent_tokenize(doc):
	sent_text = nltk.sent_tokenize(doc) # this gives you a list of sentences
	return sent_text

def word_tokenize(sent):
	tokenized_text = nltk.word_tokenize(sent)  # this gives you a list of words
	tokenized_text = [token.lower() for token in tokenized_text]  # optional: convert all words to lower case
    # tokenized_text = [token.lower() for token in tokenized_text if token not in stop_words]  # optional: remove stop words
	return tokenized_text

def pos_tag(tokenized_text):
	# POS tagging. Input is tokenized text
	tagged = nltk.pos_tag(tokenized_text)
	return tagged


def build_vocab(corpus):
    """
    Builds a vocabulary mapping from word to index based on the corpus.
    Input: list of all samples in the training data
    Return: OrderedDict - vocabulary mapping from word to integer.
    """
    # Build vocabulary
    corpus_2d = []  # convert 3d corpus to 2d list
    for doc in corpus:
    	for sent in doc:
    		corpus_2d.append(sent)
    word_counts = Counter(itertools.chain(*corpus_2d))
    # Mapping from index to word (type: list)
    vocabulary = ['<PAD/>', '<UKN/>']   # 0 for padding, 1 for unknown words
    vocabulary = vocabulary + [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocab2int = OrderedDict({x: i for i, x in enumerate(vocabulary)})
    return vocab2int

def pad_sent(sentences, max_words, max_sents):
	"""
	Pads sequences to the same length.
	Input: sentences - List of lists, where each element is a sequence.
					 - max_words: Int, maximum length of all sequences.
	"""
	
	# pad sentences in a doc
	sents_padded = pad_sequences(sentences, maxlen=max_words, padding='post') 
	# pad a doc to have equal number of sentences
	if len(sents_padded) < max_sents:
		doc_padding = np.zeros((max_sents-len(sents_padded),max_words), dtype = int)
		sents_padded = np.append(doc_padding, sents_padded, axis=0)
	else:
		sents_padded = sents_padded[:max_sents]

	return sents_padded


def getWordIdx(token, vocab2int):
    """Returns from the vocab2int the word index for a given token"""
    if token in vocab2int:
        return vocab2int[token]
    elif token.lower() in vocab2int:
        return vocab2int[token.lower()]
    return vocab2int['<UKN/>']


def build_input_data(corpus, vocab2int, max_words, max_sents):
    """
    Maps words in the corpus to integers based on a vocabulary.
    Also pad the sentences and documents into fixed shape
    Input: corpus - list of samples, each sample is a list of sentences, each sentence is a list of words
    """
    corpus_int = [[[getWordIdx(word, vocab2int) for word in sentence]for sentence in sample] for sample in corpus]
    corpus_padded = []
    for doc in corpus_int:
    	corpus_padded.append(pad_sent(doc, max_words, max_sents))
    corpus_padded = np.array(corpus_padded)    
    return corpus_padded


def load_embedding_matrix(embed_path, vocab2int, EMBEDDING_DIM, embed_type='glove'):
	"""
	return embedding_matrix 
	embedding_matrix[i] is the embedding for 'vocab2int' integer index i
	"""
	embeddings = {}
	embeddings['<PAD/>'] = np.zeros(EMBEDDING_DIM) # Zero vector for '<PAD/>' word
	embedding_UKN = np.random.uniform(-0.10, 0.10, EMBEDDING_DIM)  # Vector of small random numbers for unknown words
	# embedding_UKN = vector / np.linalg.norm(embedding_UKN)   # Normalize to unit vector
	embeddings['<UKN/>'] = embedding_UKN


	if embed_type == 'word2vec': 
		"""Loads 300x1 word vecs from Google (Mikolov) word2vec: GoogleNews-vectors-negative300.bin"""
		with open(embed_path, "rb") as f:
			header = f.readline()
			vocab_size, layer1_size = map(int, header.split())
			binary_len = np.dtype('float32').itemsize * layer1_size
			for line in range(vocab_size):
				word = []
				while True:
					ch = f.read(1)
					if ch == ' ':
						word = ''.join(word)
						break
					if ch != '\n':
						word.append(ch)
				word = word.decode('utf-8', 'ignore')
				embeddings[word] = np.fromstring(f.read(binary_len), dtype='float32')
				continue
	else:
		# load Glove or Dependency-based word embeddings
		f = open(embed_path)
		for line in f:
		    values = line.split()
		    word = values[0]
		    coefs = np.asarray(values[1:], dtype='float32')
		    embeddings[word] = coefs
		f.close()

	embedding_matrix = np.zeros((len(vocab2int) , EMBEDDING_DIM))
	for word, i in vocab2int.items():
		embedding_vector = embeddings.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
		else:   # word is unknown
			embedding_vector = np.random.uniform(-0.10, 0.10, EMBEDDING_DIM)  # Vector of small random numbers for unknown words
			# embedding_vector = vector / np.linalg.norm(embedding_vector)   # Normalize to unit vector
			embedding_matrix[i] = embedding_vector

	return embedding_matrix

def load_embedding_matrix_gensim(embed_path, vocab2int, EMBEDDING_DIM):	
	"""
	load Word2Vec using gensim: 300x1 word vecs from Google (Mikolov) word2vec: GoogleNews-vectors-negative300.bin
	return embedding_matrix 
	embedding_matrix[i] is the embedding for 'vocab2int' integer index i
	"""
	word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(embed_path, binary=True)
	embeddings = {}
	embeddings['<PAD/>'] = np.zeros(EMBEDDING_DIM) # Zero vector for '<PAD/>' word
	embedding_UKN = np.random.uniform(-0.10, 0.10, EMBEDDING_DIM)  # Vector of small random numbers for unknown words
	# embedding_UKN = vector / np.linalg.norm(embedding_UKN)   # Normalize to unit vector
	embeddings['<UKN/>'] = embedding_UKN

	for word in word2vec_model.vocab:
		embeddings[word] = word2vec_model[word]

	embedding_matrix = np.zeros((len(vocab2int) , EMBEDDING_DIM))
	for word, i in vocab2int.items():
		embedding_vector = embeddings.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
		else:   # word is unknown
			embedding_vector = np.random.uniform(-0.10, 0.10, EMBEDDING_DIM)  # Vector of small random numbers for unknown words
			# embedding_vector = vector / np.linalg.norm(embedding_vector)   # Normalize to unit vector
			embedding_matrix[i] = embedding_vector

	return embedding_matrix


# function for calculating precision and recall for each class
def getPrecision(pred_test, yTest, targetLabel):
    # reverse input argument pred_test yTest to get recall
    targetLabelCount = 0
    correctTargetLabelCount = 0

    for idx in range(len(pred_test)):
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1

            if pred_test[idx] == yTest[idx]:
                correctTargetLabelCount += 1

    if correctTargetLabelCount == 0:
        return 0

    return float(correctTargetLabelCount) / targetLabelCount










