from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sklearn.metrics import accuracy_score, \
    f1_score, \
    precision_score, \
    recall_score


def print_config(config):

    print(" |   MODEL:          | %s |" % config.__class__.__name__)

    keys = [x for x in dir(config) if not x.startswith('__')]

    for x in keys:
        print(" | %-*s | %-*s |" % (16, x, 8, str(getattr(config, x))[:6]))


def evaluate(p, y, print_it=True):

    precision = precision_score(y, p, average="samples")
    recall = recall_score(y, p, average="weighted")
    f1 = f1_score(y, p, average="weighted")
    accuracy = accuracy_score(y, p)

    if print_it:
        print("F1 Score: ", f1)
        print("Recall: ", recall)
        print("Precision: ", precision)
        print("Accuracy: ", accuracy)

    return [f1, recall, precision, accuracy]


def evaluate_multilabel(predictions, y_data, threshold, print_it = True):
    for pred in predictions:
        pred[pred > threshold] = 1
        pred[pred < threshold] = 0

    p = np.array([x.astype(int).tolist() for x in predictions])
    y = np.array([x.astype(int).tolist() for x in y_data])

    if print_it:
        print("Threshold: %.6f" % threshold)

    return evaluate(p, y, print_it)


def evaluate_multiclass(predictions, y_data, print_it=True):

    y = np.array([x.astype(int).tolist() for x in y_data])
    pred = [[0 for x in range(len(y[0]))] for z in range(len(y))]
    print(predictions.shape)
    print(len(pred))
    for index in range(0, predictions.shape[0]):
        pred[index][predictions[index].argmax()] = 1

    p = np.array([x for x in pred])

    return evaluate(p, y, print_it)


def make_loss_graph(CONST):
    from util.utils import load_pickle as load
    gru = load(CONST.DATA_DIR + 'GRU')
    lstm = load(CONST.DATA_DIR + 'LSTM')
    birnn = load(CONST.DATA_DIR + 'BiRNN')

    cutoff = 105

    gru = gru[:cutoff]
    lstm = lstm[:cutoff]
    birnn= birnn[:cutoff]

    import matplotlib.pyplot as plt
    x = [i for i in range(cutoff)]
    plt.plot(np.array(x), np.array(gru))
    plt.plot(np.array(x), np.array(lstm))
    plt.plot(np.array(x), np.array(birnn))
    plt.legend(['GRU', 'LSTM', 'BiRNN LSTM'], loc='upper right')
    plt.savefig("losses_all.png")
    print("all losses plot saved")

def re_evalute_slot3(CONST):

    from util.utils import load_pickle as load
    birnn = load(CONST.DATA_DIR + "BiRNNslot3" + "_predictions")
    evaluate_multiclass(birnn["predictions"], birnn["y"], True)

def re_evalute(CONST):
    threshold = float(CONST.THRESHOLD)

    from util.utils import load_pickle as load
    gru = load(CONST.DATA_DIR + "GRU" + "predictions")
    lstm = load(CONST.DATA_DIR + "LSTM" + "predictions")
    biRNN = load(CONST.DATA_DIR + "BiRNN" + "predictions")

    print("###########  GRU ###############")
    find_best_slot1("gru", gru["predictions"], gru["y"])
    print("########### LSTM ##############")
    find_best_slot1("lstm", lstm["predictions"], lstm["y"])
    print("########### BiRNN LSTM ##############")
    find_best_slot1("birnn", biRNN["predictions"], biRNN["y"])

def find_best_slot1(name, pred, y):

    max_threshhold_result = 0
    score = 0
    results = {}
    f1 = []
    recall = []
    precision = []
    thresholds = np.arange(0.04001, 0.20000, 0.001)
    for t in thresholds:
        pred_copy = np.copy(pred)
        t_result = evaluate_multilabel(pred_copy, y, t, print_it=False)

        if t_result[0] > score:
            score = t_result[0]
            max_threshhold_result = t
            results = t_result

        f1.append(t_result[0])
        recall.append(t_result[1])
        precision.append(t_result[2])

    print("Threshold: %.6f" % max_threshhold_result)
    print("F1 Score: ", results[0])
    print("Recall: ", results[1])
    print("Precision: ", results[2])

    import matplotlib.pyplot as plt
    plt.plot(np.array(thresholds), np.array(f1), '--')
    plt.plot(np.array(thresholds), np.array(recall))
    plt.plot(np.array(thresholds), np.array(precision), ':')
    plt.legend(['F1', 'Recall', 'Precision'], loc='upper right')
    plt.savefig("score_" + name + ".png")
    try:
        plt.close()
    except AttributeError as e:
        pass
    print("Saved BiRNN F1, Recall, and Precision")

if __name__ == "__main__":

    import argparse
    from read.config_reader import CONST
    CONST.parse_argument(argparse.ArgumentParser())

    #make_loss_graph(CONST)

    re_evalute(CONST)

    # slot 3
    re_evalute_slot3(CONST)
