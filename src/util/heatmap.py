import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np



def heatmap(p, sentence, target, path):

    fig, ax = plt.subplots()
    p = np.asarray([p[:len(sentence)]])
    ax = sns.heatmap(p, xticklabels=sentence, cmap=plt.cm.Reds, cbar=False)
    plt.xticks(rotation=30)
    ax.yaxis.set_visible(False)
    ax.set_title('target: ' + target)
    plt.savefig(path)
    plt.close()


def avg_distance_and_heatmaps(alphas, sentences, path):
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    sentence = 0
    counted_sentences = 0
    total_distance = 0
    for s in sentences:
        #if total_distance > 10:
        #    break
        if len(s.aspect_targets) > 0:
            target = tokenizer.tokenize(s.aspect_targets[0].target)[0].lower()
            if len(s.uni_grams) > np.argmax(alphas[sentence]) and target != 'null':
                distance = abs(s.uni_grams.index(target) - np.argmax(alphas[sentence]))
                total_distance += distance
                heatmap(alphas[sentence],
                        s.uni_grams,
                        target,
                        path + str(sentence))
                counted_sentences += 1
            sentence += 1
    print("Avg Distace: ", total_distance/counted_sentences)
    #print(total_distance, counted_sentences)
