import _pickle as cPickle

class_dict = {"Positive": 0, "positive": 0, "Negative": 1, "negative": 1, "none": 2, "None": 2}
class_dict_reverse = {0: "Positive", 1: "Negative", 2: "None"}


def save_pickle(file, data):
    fp = open(file + '.pickle', 'wb')
    cPickle.dump(data, fp, protocol=2)
    fp.close()


def load_pickle(file):
    import os
    assert os.path.isfile(file + '.pickle'), "file doesn't exist"
    fp = open(file + '.pickle', 'rb')
    data = cPickle.load(fp)
    fp.close()
    return data


def check_saved_file(file):
    import os
    return os.path.exists(file + '.pickle') and os.stat(file + '.pickle').st_size > 0


def result_html(dev_sentences, dev_true, dev_hat, out_name):
    dev_true_classes = [class_dict_reverse[d] for d in dev_true]
    dev_hat_classes = [class_dict_reverse[d] for d in dev_hat]
    res_data = [(s.text, y_true, y_hat) for (s, y_true, y_hat) in zip(dev_sentences, dev_true_classes, dev_hat_classes)]
    corrects = [(s, y_true, y_hat) for (s, y_true, y_hat) in res_data if y_true == y_hat]
    errors = [(s, y_true, y_hat) for (s, y_true, y_hat) in res_data if y_true != y_hat]
    html = "<html><body>"
    html += "<table border='1'>"
    for error in errors:
        html += "<tr>"
        html += "<td>" + error[0] + "</td>"
        html += "<td><font style=\"color:green\">" + error[1] + "</font></td>"
        html += "<td><font style=\"color:red\">" + error[2] + "</font></td>"
        html += "</tr>"
    html += "</table>"
    html += "<br><hr><br>"
    html += "<table border='1'>"
    for correct in corrects:
        html += "<tr>"
        html += "<td>" + correct[0] + "</td>"
        html += "<td><font style=\"color:green\">" + correct[1] + "</font></td>"
        html += "<td><font style=\"color:green\">" + correct[2] + "</font></td>"
        html += "</tr>"
    html += "</table>"
    html += "</body></html>"
    # print(html)
    f = open(out_name + ".html", 'w')
    f.write(html)
    f.close()
    print(out_name + ".html")
