"""
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- batch_size - the batch size

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from read.data_processing import semeval_itterator

from read.config_reader import CONST
from util.utils import load_pickle as load
from util.utils import save_pickle
from math import ceil

import datetime
import argparse

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)

tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


def variable_summaries(variable, name):
    """Attach a lot of summaries to a Tensor"""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(variable)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(variable - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(variable))
        tf.scalar_summary('min/' + name, tf.reduce_min(variable))
        tf.histogram_summary(name, variable)


class CNNModel(object):
    """CNN model."""

    def __init__(self, is_training, config):
        """

        :rtype: model to train or evaluate
        """
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        vocab_size = config.vocab_size

        # inputs
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])  # 20 x 70 x [100]
        self._targets = tf.placeholder(tf.int32, [batch_size, config.classes])  # 20 x 13

        # Embeddings
        self.embedding = tf.get_variable("embedding",
                                         [vocab_size, config.embedding_size],
                                         dtype=data_type(),
                                         trainable=False)
        inputs_embedded = tf.nn.embedding_lookup(self.embedding, self._input_data)

        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        self._embedded_expanded = tf.expand_dims(inputs_embedded, -1)
        self._output, num_filters_total = self.getCNN(config, self._embedded_expanded, initializer)

        if config.get_summary:
            variable_summaries(self._output, config.__class__.__name__)
            # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W",
                                shape=[num_filters_total, config.classes],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                name="sentiment_weight")
            b = tf.get_variable("b", [config.classes], dtype=data_type(),
                                initializer=initializer,
                                name="sentiment_b")
            self._scores = tf.nn.xw_plus_b(self._output, W, b, name="scores")
            self._predictions = tf.argmax(self._scores, 1, name="predictions")

            # Accuracy
        with tf.name_scope("accuracy"):
            self._correct_predictions = tf.equal(self._predictions, tf.argmax(self._targets, 1))
            self._accuracy = tf.reduce_mean(tf.cast(self._correct_predictions, "float"), name="accuracy")

        if not is_training:
            return

        l2_loss = tf.constant(0.0)
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self._scores, tf.to_float(self._targets))
            self._loss = self._cost = tf.reduce_mean(losses) + config.l2_reg_lambda * l2_loss

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        global_step = tf.Variable(0, trainable=False)

        learning_rate = tf.train.exponential_decay(
          config.learning_rate,             # Base learning rate.
          global_step * config.batch_size,  # Current index into the dataset.
          5000,                             # Decay step.
          0.85,                             # Decay rate.
          staircase=True)
        # Define Training procedure
        # global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(self._loss)
        self._optimizer = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


    def getCNN(self, config, inputs, initializer):
        # Create a convolution + maxpool layer for each filter size
        filter_sizes = list(map(int, config.filter_sizes.split(",")))
        pooled_outputs = []
        stride = 1
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, config.embedding_size, 1, config.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W-%s" % filter_size, dtype=data_type())
                b = tf.Variable(tf.truncated_normal([config.num_filters], stddev=0.1), name="b-%s" % filter_size, dtype=data_type())
                #b = tf.get_variable("b", [config.num_filters], dtype=data_type(), initializer=initializer)
                conv = tf.nn.conv2d(
                        inputs,
                        W,
                        strides=[1, stride, stride, 1],
                        padding="VALID",
                        name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.avg_pool(h,
                    ksize=[1, ceil((config.num_steps - filter_size + 1)/stride), 1, 1],
                    strides=[1, stride, stride, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = config.num_filters * len(filter_sizes)
        h_pool = tf.concat(3, pooled_outputs)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, config.drop_out_keep)

        return h_drop, num_filters_total


    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def merged_summary(self):
        return self._merged_summary

    @property
    def output(self):
        return self._output

    @property
    def logits(self):
        return self._logits

    @property
    def predictions(self):
        return self._predictions

    @property
    def correct_predictions(self):
        return self._correct_predictions

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def loss(self):
        return self._loss

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def scores(self):
        return self._scores


class CNN(object):

    batch_size = 10
    classes = 3
    drop_out_keep = 0.7
    embedding_size = 100
    filter_sizes = '2,3,4,5'
    get_summary = True
    init_scale = 0.15
    l2_reg_lambda = 0.0
    learning_rate = 0.0001
    max_max_epoch = 100
    max_sequence = 70
    num_steps = 12
    num_filters = 128
    vocab_size = 10000


def run_epoch(session, m, x_data, y_data, writer=None, run_options=None, run_metadata=None, verbose=False,
              category=False):
    """Runs the model on the given data.
    :param session:
    :param m:
    :param y_data:
    :param x_data:
    :param eval_op: REMOVED!!!
    :param verbose:
    """
    epoch_size = ((len(x_data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    epsilon = 1e-8
    delta_cost = 0.5
    prev_cost = 0.0

    losses = []

    merged = tf.merge_all_summaries()
    for step, (x, y, l, e, a) in enumerate(semeval_itterator(x_data,
                                                          y_data,
                                                          m.batch_size,
                                                          m.num_steps, category=category)):
        # if delta_cost < epsilon:
        # print("delta: ", delta_cost, " epsilon: ", epsilon)
        # break

        if writer:
            cost, loss, summary, _ = session.run([m.cost, m.loss, merged, m.optimizer],
                                                        {m.input_data: x,
                                                         m.targets: y},
                                                        options=run_options,
                                                        run_metadata=run_metadata)
        else:
            cost, loss, _ = session.run([m.cost, m.loss, m.optimizer],
                                        {m.input_data: x,
                                         m.targets: y})

        # writer.add_run_metadata(run_metadata, 'step%03d' % step)
        if writer:
            writer.add_summary(summary, step)

        delta_cost = abs(cost - prev_cost)
        prev_cost = cost
        costs += cost
        iters += m.num_steps
        losses.append(loss)

        # print("iterations: %d cost %.4f loss %.6f" % (iters, cost, loss))
        # print("updating?", w)

        #if verbose and iters % (m.batch_size * 5) == 0:
        #    print("step %.3f loss : %.6f speed: %.0f wps" %
        #          (step * 1.0 / epoch_size, loss, iters * m.batch_size / (time.time() - start_time)))

    return np.exp(abs(costs) / iters), loss, losses


def get_config(model):
    if model == "CNN":
        return CNN()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(CONST, data):
    config = get_config("CNN")

    # if not CONST.TRAIN:
    #    config.num_steps = 1

    config.vocab_size = len(data["embeddings"])
    config.max_max_epoch = CONST.MAX_EPOCH
    #from util.evaluations import print_config
    #print_config(config)

    tf.reset_default_graph()
    # start graph and session
    # config=tf.ConfigProto(log_device_placement=True) pass to tf.Session
    # to see which devices all operations can run on
    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:

        with tf.variable_scope("model", reuse=None, initializer=tf.contrib.layers.xavier_initializer()):
            training_model = CNNModel(is_training=True, config=config)  # model class

        if CONST.TRAIN:

            tf.initialize_all_variables().run()
            session.run(training_model.embedding.assign(data["embeddings"]))  # train

            # Reload save epoch training time
            if CONST.RELOAD_TRAIN:
                print("Reloading previous run...")
                saver = tf.train.Saver()
                saver.restore(sess=session, save_path=CONST.CNN_MODEL_PATH + "cnn")

            all_losses = []

            train_writer = run_metadata = run_options = False

            if config.get_summary:
                # session = tf.InteractiveSession()
                train_writer = tf.train.SummaryWriter(
                    CONST.SLOT1_MODEL_PATH + "attention_graph/" + config.__class__.__name__,
                    session.graph)

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            for i in range(config.max_max_epoch):  # setup epoch

                train_perplexity, loss, losses = run_epoch(session,
                                                           training_model,
                                                           data["x_train"],
                                                           data["p_train"],
                                                           train_writer,
                                                           run_options,
                                                           run_metadata,
                                                           category=data["a_train"],
                                                           verbose=True)
                all_losses = all_losses + [np.mean(losses)]

                print("Epoch: %d Avg. Cost: %.6f" % (i + 1, np.mean(all_losses)))

            from util.evaluations import print_config
            print_config(config)

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            # plt.plot([np.mean(all_losses[i-50:i]) for i in range(len(all_losses))])
            figure_name = CONST.OUT_DIR + "loss/" + "losses_slot3" + config.__class__.__name__ + ".png"
            x = [i for i in range(len(all_losses))]
            plt.plot(np.array(x), np.array(all_losses))
            plt.savefig(figure_name)
            save_pickle(CONST.DATA_DIR + config.__class__.__name__ + "_slot3", all_losses)
            print("saved slot3 losses.png and losses data", figure_name)
            print("loss: ", figure_name)
            print("loss data: ", CONST.DATA_DIR + config.__class__.__name__ + "_slot3" + ".pickle")

            #saver = tf.train.Saver(tf.all_variables())
            #path = saver.save(sess=session, save_path=CONST.CNN_MODEL_PATH + "cnn")
            #print("model saved: " + path)

            if config.get_summary:
                train_writer.close()

                # session.close()  # doesn't seem to close under scope??

        if not False:
            with tf.variable_scope("model", reuse=True):  # reuse scope to evaluate model :-)


                # Initialize Model Graph
                validation_model = training_model

                predictions = []
                correct_predictions = []
                # Get Predictions
                for step, (x, y, l, e, a) in enumerate(semeval_itterator(data["x_test"],
                                                                      data["p_test"],
                                                                      validation_model.batch_size,
                                                                      validation_model.num_steps,
                                                                      category=data["a_train"],
                                                                      shuffle_examples=False)):

                    pred, scores = session.run([validation_model.predictions,
                                                  validation_model.scores],
                                                 {validation_model.input_data: x,
                                                  validation_model.targets: y})

                    predictions = predictions + pred.tolist()
                    correct_predictions = correct_predictions + scores.tolist()

                even_batch = len(data["x_test"]) % config.batch_size
                remove_added_batch = config.batch_size - even_batch
                predictions = correct_predictions
                del predictions[-remove_added_batch:]

                predictions = np.asarray(predictions)

                from util.evaluations import print_config

                from util.evaluations import evaluate_multiclass

                y = [np.asarray(e) for e in data["p_test"]]

                save_pickle(CONST.DATA_DIR + config.__class__.__name__ + "slot3_predictions",
                            {"predictions": predictions, "y": y})

                evaluate_multiclass(predictions, y, True)

                print("predictions saved")
                # session.close()  # doesn't seem to close under scope??

if __name__ == "__main__":
    # Set and Overload Arguments
    CONST.parse_argument(argparse.ArgumentParser())

    # Set Time of Experiment
    now = datetime.datetime.now()
    time_stamp = "_".join([str(a) for a in [now.month, now.day, now.hour, now.minute, now.second]])

    data = load(CONST.DATA_DIR + CONST.DATA_FILE)

    """
        x_train = data["x_train"]
        x_dev = data["x_dev"]
        x_test = data["x_test"]
        y_train = data["y_train"]
        y_dev = data["y_dev"]
        y_test = data["y_test"]
        l_train = data["l_train"]
        l_dev = data["l_dev"]
        l_test = data["l_test"]
        train_sentences = data["train_sentences"]
        dev_sentences = data["dev_sentences"]
        test_sentences = data["test_sentences"]
        embeddings = data["embeddings"]
        aspects = data["aspects"]
    """

    # tf.app.run(CONST, data)
    main(CONST, data)
