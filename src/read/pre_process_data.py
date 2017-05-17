from __future__ import print_function
from read import \
    data_processing as dp, \
    vectorize_data as vd

from read.config_reader import CONST

from util.utils import save_pickle as save

import datetime
import argparse

if __name__ == "__main__":

    # Set and Overload Arguments
    CONST.parse_argument(argparse.ArgumentParser())

    # Set Time of Experiment
    now = datetime.datetime.now()
    time_stamp = "_".join([str(a) for a in [now.month, now.day, now.hour, now.minute, now.second]])
    name = time_stamp

    corpa = dp.get_semeval_data(CONST)

    train_data = corpa['restaurants']['train']['corpus']
    dev_data   = corpa['restaurants']['dev']['corpus']
    test_data  = corpa['restaurants']['test']['corpus']

    # Send all opinions for training
    # Multi-Label Classification
    x_train, x_dev, x_test, \
    y_train, y_dev, y_test, \
    l_train, l_dev, l_test, \
    train_sentences, dev_sentences, test_sentences, \
    a_train, a_dev, a_test, \
    p_train, p_dev, p_test, \
    n_train, n_dev, n_test, \
    embeddings, aspects = vd.vectorize_rnn(train_data.corpus,
                                           dev_data.corpus,
                                           test_data.corpus,
                                           CONST)

    data = {"x_train": x_train, "x_dev": x_dev, "x_test": x_test,
            "y_train": y_train, "y_dev": y_dev, "y_test": y_test,
            "l_train": l_train, "l_dev": l_dev, "l_test": l_test,
            "a_train": a_train, "a_dev": a_dev, "a_test": a_test,
            "p_train": p_train, "p_dev": p_dev, "p_test": p_test,
            "n_train": n_train, "n_dev": n_dev, "n_test": n_test,
           "train_sentences": train_sentences, "dev_sentences": dev_sentences, "test_sentences": test_sentences,
            "embeddings": embeddings, "aspects": aspects}

    save(CONST.DATA_DIR + CONST.DATA_FILE, data)
