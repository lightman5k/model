import numpy as np
import re
import cPickle
import itertools
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def copy_words(A, B):
    A = A + ' '
    return A * B


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


def load_data_and_labels(filepath):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(filepath).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    # Split by words
    x_text = [s.split(" ") for s in positive_examples]
    # Generate labels
    return x_text


def pad_sentences(sentences, sequence_length=None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    padding_word = "<PAD/>"

    length = [len(x) for x in sentences]
    if sequence_length == None:
        sequence_length = int(np.percentile(length, 90))

    print('max sentence length is %d' % sequence_length)
    padded_sentences = []
    sentence_length = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        if num_padding > 0:
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
            sentence_length.append(len(sentence))
        else:
            padded_sentences.append(sentence[:sequence_length])
    return [sequence_length, padded_sentences, sentence_length]


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return x


def split_num(sent):
    # split number followed by word character
    sent_new = []
    for word in sent:
        match = re.match(r"([0-9]+)([a-zA-Z]+)", word, re.I)
        if match:
            sent_new.append(match.group(1))
            sent_new.append(match.group(2))
        else:
            sent_new.append(word)
    return sent_new


def load_data(corpus, sequence_length=None):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # sentences = load_data_and_labels(filepath)

    x_text = [s.strip() for s in corpus]
    sentences = [clean_str(sent) for sent in x_text]
    sentences = [s.split(" ") for s in sentences]
    sentences = [split_num(s) for s in sentences]

    sequence_length, sentences_padded, sen_length = pad_sentences(sentences, sequence_length)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x = build_input_data(sentences_padded, vocabulary)
    return [x, sequence_length, vocabulary, vocabulary_inv]


def make_bow(corpus, bow_file):
    if os.path.exists(bow_file):
        a = cPickle.load(open(bowpath, "rb"))
        c_vocab, c_vector = a[0], a[1]
    else:
        c_dict = TfidfVectorizer(lowercase=False, stop_words='english', min_df=2)
        c_vector = c_dict.fit_transform(corpus)
        c_vocab = c_dict.get_feature_names()
        cPickle.dump([c_vocab, c_vector], open(bow_file, "wb"))

    return c_vocab, c_vector


def reconstrut_data(filepath, corppath):
    x = cPickle.load(open(filepath, "rb"))
    c_vocab, c_vector = x[0], x[1]
    c_vector = c_vector.todense()
    tfile = open(corppath, 'w')
    for c in c_vector:
        x = np.squeeze(np.asarray(c))
        new_string = ''.join([copy_words(c_vocab[i], x[i]) for i in np.flatnonzero(x > 0)])
        tfile.write("%s\n" % new_string[:-1])
    tfile.close()


def load_bin_vec(fname, svocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    word_vecs["<PAD/>"] = np.zeros(300)
    num_found1 = 0
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word = word.decode('utf-8', 'ignore')
            if word in svocab:
                num_found1 += 1
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                continue
            else:
                f.read(binary_len)
    print('%d out of %d words found in the word2vec vocab' % (num_found1, len(svocab)))

    for i, word in enumerate(svocab):
        if word not in word_vecs:
            # print(word)
            word_vecs[word] = np.random.uniform(-0.1, 0.1, 300)

    return word_vecs


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