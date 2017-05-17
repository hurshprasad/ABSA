from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from os import listdir, path
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import util.file_utils as file_util
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from random import shuffle

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

import numpy as np


class Opinion:
    expression = ""
    expression_begin = -1
    expression_end = -1
    aspect = ""
    location = ""
    location_begin = -1
    location_end = -1
    sentiment = ""

    def __init__(self, expression, expression_begin, expression_end, aspect, location, location_begin, location_end,
                 sentiment):
        self.expression = expression
        self.aspect = aspect
        self.location = location
        self.sentiment = sentiment
        self.expression_begin = expression_begin
        self.expression_end = expression_end
        self.location_begin = location_begin
        self.location_end = location_end


def get_semeval_data(c):
    corpora = dict()
    corpora['restaurants'] = dict()

    import xml.etree.ElementTree as ET
    from read.corpus_opinions import Corpus

    # training data
    train_reviews = ET.parse(c.DATA_DIR + c.TRAIN_FILES[0]).getroot().findall('Review') + \
                    ET.parse(c.DATA_DIR + c.TRAIN_FILES[1]).getroot().findall('Review')

    train_sentences = []
    for r in train_reviews:
        train_sentences += r.find('sentences').getchildren()

    # Dev/Test Phase A data
    dev_reviews = ET.parse(c.TEST_DIR + c.DEV_FILES[0]).getroot().findall('Review')

    dev_sentences = []
    for r in dev_reviews:
        dev_sentences += r.find('sentences').getchildren()

    # Test Phase A GOLD data
    test_reviews = ET.parse(c.TEST_DIR + c.TEST_FILES[0]).getroot().findall('Review')

    test_sentences = []
    for r in test_reviews:
        test_sentences += r.find('sentences').getchildren()

    # TODO: parser is not loading aspect words and opinions - FIXED {hurshprasad}
    train_corpus = Corpus(train_sentences)
    dev_corpus = Corpus(dev_sentences)
    test_corpus = Corpus(test_sentences)

    corpora['restaurants']['train'] = dict()
    corpora['restaurants']['dev'] = dict()
    corpora['restaurants']['test'] = dict()

    corpora['restaurants']['train']['corpus'] = train_corpus
    corpora['restaurants']['dev']['corpus'] = dev_corpus
    corpora['restaurants']['test']['corpus'] = test_corpus

    return corpora


def text_to_token_lemmas(text):
    text = text.lower().replace("n't", " not")
    text = text.lower().replace("ain't", " is not")
    text = text.lower().replace("aint", " is not")
    text = text.lower().replace("wasnt", " was not")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    lemmas = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens]
    return lemmas, tokens


def read_json(file):
    sentences = []
    if path.isfile(file):
        with open(file, 'r') as myfile:
            json_txt = myfile.read()
            json_dicts = json.loads(json_txt)
            for json_dict in json_dicts:
                sentence = AnnotatedSentence.from_json_dict(json_dict)
                sentences.append(sentence)
    else:
        print("file " + file + " Not Found!!")
    return sentences


def stat(data, title):
    opinions = [op for op_list in [s.opinions for s in data] for op in op_list]
    print(title + ": " + str(len(opinions)))
    aspects = ["general", "price", "transit-location", "safety", "live",
               "nightlife"]  # np.unique(np.array([op.aspect for op in opinions]))
    pos = []
    neg = []
    for aspect in aspects:
        pos_aspect = [d.sentiment for d in opinions if d.aspect == aspect and d.sentiment == "Positive"]
        neg_aspect = [d.sentiment for d in opinions if d.aspect == aspect and d.sentiment == "Negative"]
        pos.append(len(pos_aspect))
        neg.append(len(neg_aspect))

    N = len(aspects)
    # sns.set_style("darkgrid")
    # ind = np.arange(N)  # the x locations for the groups
    # width = 0.35  # the width of the bars: can also be len(x) sequence
    #
    # p1 = plt.bar(ind, pos, width, color='g')
    # p2 = plt.bar(ind, neg, width, color='r', bottom=pos)
    #
    # plt.ylabel('Counts')
    # plt.title('Number of Sentences For Corresponding Aspects - ' + title)
    # plt.xticks(ind + width / 2., (aspects))
    # plt.legend((p1[0], p2[0]), ('Positive', 'Negative'))
    #
    # plt.show()


def split_data(data, file_prefix):
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)

    total_len = len(data)
    portion_size = int(total_len / 7)

    dev_data = data[0:portion_size]
    test_data = data[portion_size: portion_size * 3]
    train_data = data[portion_size * 3:]

    file_util.write_to_file(file_prefix + "_train.json", sentences_to_json(train_data))
    file_util.write_to_file(file_prefix + "_dev.json", sentences_to_json(dev_data))
    file_util.write_to_file(file_prefix + "_test.json", sentences_to_json(test_data))

    return train_data, dev_data, test_data


def sentences_to_json(sentences):
    sentences_dict = []
    s_id = 0
    for sentence in sentences:
        s_id += 1
        sentence_dict = {'id': sentence.id, 'irrelevant': False, 'uncertain': False, 'path': sentence.path,
                         'category': sentence.category}
        if len(sentence.relevant_text) > 0:
            sentence_dict['text'] = sentence.relevant_text
        elif len(sentence.text) > 0:
            sentence_dict['text'] = sentence.text
        ops = sentence.opinions
        opinions = []
        for opinion in ops:
            opinion_dict = {}
            opinion_dict['location'] = opinion.location
            opinion_dict['aspect'] = opinion.aspect
            opinion_dict['sentiment'] = opinion.sentiment
            opinion_dict['location_begin'] = opinion.location_begin
            opinion_dict['location_end'] = opinion.location_end
            opinion_dict['expression'] = opinion.expression
            opinion_dict['expression_begin'] = opinion.expression_begin
            opinion_dict['expression_end'] = opinion.expression_end
            opinions.append(opinion_dict)
        sentence_dict['opinions'] = opinions
        sentences_dict.append(sentence_dict)
    # convert to json
    json_ser = json.dumps(sentences_dict)
    return json_ser


def split_data_ids(data, dir):
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)

    total_len = len(data)
    portion_size = int(total_len / 7)

    dev_data = data[0:portion_size]
    test_data = data[portion_size: portion_size * 3]
    train_data = data[portion_size * 3:]

    train_ids = "\n".join([d.id for d in train_data])
    dev_ids = "\n".join([d.id for d in dev_data])
    test_ids = "\n".join([d.id for d in test_data])

    file_util.write_to_file(dir + "single_train.ids", train_ids)
    file_util.write_to_file(dir + "single_dev.ids", dev_ids)
    file_util.write_to_file(dir + "single_test.ids", test_ids)

    return train_data, dev_data, test_data


def read_single_location_json_data():
    dir = "/Users/marziehsaeidi/Documents/Apps/UrbanScala/data/aspect/brat/output/"
    data = read_json(dir + "generation_single_all.json")
    return data


def read_split_data(mod, dir="/Users/marziehsaeidi/Documents/Apps/naga/naga/members/marzieh/paper/data/"):
    train_file_name = dir + mod + "_train.ids"
    dev_file_name = dir + mod + "_dev.ids"
    test_file_name = dir + mod + "_test.ids"
    train_file = open(train_file_name)
    train_ids = [id for id in train_file.read().split("\n") if len(id) > 0]
    dev_file = open(dev_file_name)
    dev_ids = [id for id in dev_file.read().split("\n") if len(id) > 0]
    test_file = open(test_file_name)
    test_ids = [id for id in test_file.read().split("\n") if len(id) > 0]

    data = read_json(dir + mod + ".json")
    train_data = [d for d in data if d.id in train_ids]
    dev_data = [d for d in data if d.id in dev_ids]
    test_data = [d for d in data if d.id in test_ids]
    # for sent in train_data:
    #     ops = sent.opinions
    #     aspect_ops = [op for op in ops if op.aspect == aspect]
    #     if len(aspect_ops) > 0:
    #         print(sent.text)

    return train_data, dev_data, test_data


def read_data_from_files(train_files, dev_files, test_files, dir="", train_files_percents=[]):
    train_sentences = []
    if len(train_files_percents) == 0:
        train_files_percents = [1 for t in train_files]
    for file, percent in zip(train_files, train_files_percents):
        file_data = read_json(dir + file)
        shuffle(file_data)  ## TODO: Any value to shuffling
        take = int(percent * len(file_data))
        train_sentences += file_data[0:take]

    dev_sentences = []
    for file in dev_files:
        dev_sentences += read_json(dir + file)

    test_sentences = []
    for file in test_files:
        test_sentences += read_json(dir + file)

    return train_sentences, dev_sentences, test_sentences


def read_generated(name):
    data = read_json(name)
    return data


def semeval_itterator(x_data, y_data, x_length, batch_size, num_steps, shuffle_examples=True, category=False, polarity=False, target=False):

    indexer = list(range(0, len(y_data)))
    data_len = len(indexer)

    even_batch = data_len % batch_size
    add_to_indexer = batch_size - even_batch

    if shuffle_examples:
        shuffle(indexer)
    # raw_data = np.array(indexer, dtype=np.int32)

    indexer.extend([indexer[-1] for i in range(add_to_indexer)])

    data_len += add_to_indexer

    batch_len = data_len // batch_size

    for i in range(batch_len):
        x = np.asarray([x_data[indexer[n]][:num_steps] for n in list(range(i*batch_size, (i+1)*batch_size))])
        y = np.asarray([y_data[indexer[n]] for n in list(range(i*batch_size, (i+1)*batch_size))])
        l = np.asarray([x_length[indexer[n]] for n in list(range(i*batch_size, (i+1)*batch_size))])
        if type(target) is list and type(polarity) is list:
            p = np.asarray([polarity[indexer[n]] for n in list(range(i*batch_size, (i+1)*batch_size))])
            t = np.asarray([target[indexer[n]] for n in list(range(i*batch_size, (i+1)*batch_size))])
            yield (x, y, l, p, t)
        elif type(target) is list:
            t = np.asarray([target[indexer[n]] for n in list(range(i*batch_size, (i+1)*batch_size))])
            yield (x, y, l, t)
        elif type(category) is list and type(polarity) is list:
            e = np.asarray([category[indexer[n]][0] for n in list(range(i*batch_size, (i+1)*batch_size))])
            a = np.asarray([category[indexer[n]][1] for n in list(range(i*batch_size, (i+1)*batch_size))])
            p = np.asarray([polarity[indexer[n]] for n in list(range(i*batch_size, (i+1)*batch_size))])
            yield (x, y, l, p, e, a)
        elif type(category) is list:
            e = np.asarray([category[indexer[n]][0] for n in list(range(i*batch_size, (i+1)*batch_size))])
            a = np.asarray([category[indexer[n]][1] for n in list(range(i*batch_size, (i+1)*batch_size))])
            yield (x, y, l, e, a)
        else:
            yield (x, y, l)
