import numpy as np
import read.w2v_loader as w2v
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

'''
['restaurant#general', 'service#general', 'food#quality', 'drinks#style_options', 'restaurant#prices', 'ambience#general', 'restaurant#miscellaneous', 'food#style_options', 'location#general', 'drinks#quality', 'food#prices', 'drinks#prices']

['restaurant#general', 0
 'service#general', 1
 'food#quality', 2
 'drinks#style_options', 3
 'restaurant#prices', 4
 'ambience#general', 5
 'restaurant#miscellaneous', 6
 'food#style_options', 7
 'location#general', 8
 'drinks#quality', 9
 'food#prices', 10
 'drinks#prices' 11
 ]
'''

sentiments = ['positive', 'negative', 'neutral'] #, 'none']

def vectorize_rnn(train_sentences, dev_sentences, test_sentences, c=None):
    print("Vectorizing...")
    train_arr = np.array(train_sentences)
    dev_arr = np.array(dev_sentences)
    test_arr = np.array(test_sentences)
    all_sentences = np.append(np.append(train_arr, dev_arr), test_arr)

    lens = []

    # TODO all aspects and get indexes
    aspects = []

    aspects_dict = {}

    # Sentence is a type Instance
    for s in train_arr:
        if len(s.aspect_targets) < 1:
            if "NONE" in aspects_dict:
                aspects_dict["NONE"] += 1
            else:
                aspects_dict["NONE"] = 1
        for a_targets in s.aspect_targets:
            if a_targets.category not in aspects:
                aspects.append(a_targets.category)
            if a_targets.category in aspects_dict:
                aspects_dict[a_targets.category] += 1
            else:
                aspects_dict[a_targets.category] = 1

    aspects.append("NONE")

    vocab = set([])
    for s in all_sentences:
        s.get_vsm(c=c)
        lens.append(len(s.uni_grams))
        vocab = vocab.union(s.uni_grams)
        for opinion in s.aspect_targets:
            w = [opinion.entity.split('_')[0], opinion.aspect.split('_')[0]]
            vocab = vocab.union(w)

    max = 70

    model = w2v.load_vectors(c.W2VEC_FILE, list(vocab))
    embeddings = model.embeddings
    length = len(embeddings)
    ind = dict(zip(model.words, np.arange(0, length)))

    ## Adding extra tokens
    ind.update({c.TARGET_MASK: length})  # OK don't use
    ind.update({"MISSING": (length + 1)})
    ind.update({"EOS": (length + 2)})
    ind.update({"PAD": (length + 3)})
    ind.update({"DROP": (length + 4)})

    MISSING_vec = np.zeros((1, embeddings.shape[1]))
    EOS_vec = np.zeros((1, embeddings.shape[1]))
    PAD_vec = np.zeros((1, embeddings.shape[1]))
    DROP_vec = np.zeros((1, embeddings.shape[1]))

    embeddings = np.append(embeddings, MISSING_vec, axis=0)
    embeddings = np.append(embeddings, EOS_vec, axis=0)
    embeddings = np.append(embeddings, PAD_vec, axis=0)
    embeddings = np.append(embeddings, DROP_vec, axis=0)

    x_train, y_train, t_train, a_train, p_train, l_train, s_train = sentences_to_x_y_l(train_arr, ind, max, aspects, c)
    x_dev, y_dev, t_dev, a_dev, p_dev, l_dev, s_dev = sentences_to_x_y_l(dev_arr, ind, max, aspects, c)
    x_test, y_test, t_test, a_test, p_test, l_test, s_test = sentences_to_x_y_l(test_arr, ind, max, aspects, c)

    return x_train, x_dev, x_test, \
           y_train, y_dev, y_test, \
           t_train, t_dev, t_test, \
           s_train, s_dev, s_test, \
           a_train, a_dev, a_test, \
           p_train, p_dev, p_test, \
           l_train, l_dev, l_test, \
           embeddings, aspects


def sentences_to_x_y_l(all_sentences, ind, max, aspects, c):
    X = []  # indexes
    y = []  # [size 12 hot]
    l = []  # target word
    a = []  # [entity],[aspect]
    s = []  # [0 0 0 0] positive, negative, neutral, none
    le = [] # length of word
    sentences = []
    vsm_filtered_all = []

    for sentence in all_sentences:
        # vsm, left_vsm, right_vsm = sentence.get_vsm(center=location_name, config=config)
        y_ = [0] * len(aspects)
        s_ = [0] * len(sentiments)

        if len(sentence.aspect_targets) < 1:
            category_index = aspects.index("NONE")
            y_[category_index] = 1
            a.append([ind["MISSING"], ind["MISSING"]])
            s_[len(sentiments) - 1] = 1
            l.append(0)
        else:
            ea_found = False
            t_found = False
            l.append(0)
            for opinion in sentence.aspect_targets:
                category_index = aspects.index(opinion.category)
                y_[category_index] = 1

                if not t_found:
                    if opinion.target and opinion.target.lower() != "null":
                        target = tokenizer.tokenize(opinion.target.lower())
                        l[-1] = sentence.uni_grams.index(target[0].lower())
                        t_found = True

                if not ea_found:
                    a.append([ind[opinion.entity.split('_')[0]], ind[opinion.aspect.split('_')[0]]])
                    ea_found = True

                if not opinion.polarity:
                    s_[len(sentiments)-1] = 1
                else:
                    s_[sentiments.index(opinion.polarity.lower())] = 1

        s.append(s_)
        y.append(np.asarray(y_))
        le.append(len(sentence.uni_grams))
        vsm_filtered = [v if v in ind else "MISSING" for v in sentence.uni_grams]
        padding = np.append(["EOS"], np.repeat("PAD", max - len(vsm_filtered)), axis=0)
        vsm_filtered_all.append(vsm_filtered)

        all_words = np.append(vsm_filtered, padding, axis=0)
        x_int_rep = [ind[w] for w in all_words]
        X.append(np.asarray(x_int_rep))
        sentences.append(sentence)

    return X, y, l, a, s, le, sentences

