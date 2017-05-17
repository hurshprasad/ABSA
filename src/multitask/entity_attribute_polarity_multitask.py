"""
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
    - keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from read.data_processing import semeval_itterator

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

from read.config_reader import CONST
from util.file_utils import check_saved_file
from util.utils import load_pickle as load
from util.utils import save_pickle

import datetime
import argparse


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


class shared_Embeddings(object):
    """ Shared Embeddings for Multi-task """

    def __init__(self, config):
        self._input_data = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])  # 20 x 70 x [100]
        with tf.device("/cpu:0"):
            self._embeddings = tf.get_variable("embedding",
                                              [config.vocab_size, config.embedding_size],
                                               dtype=data_type(),
                                              trainable=False)

            self._inputs = tf.nn.embedding_lookup(self._embeddings, self._input_data)

    @property
    def source_data(self):
        return self._input_data

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def inputs(self):
        return self._inputs


class shared_BiRNN(object):
    """shared Multi-task RNN model."""

    def __init__(self, is_training, config, inputs):
        # Need scope for each direction here: https://github.com/tensorflow/tensorflow/issues/799
        with tf.variable_scope('forward_multi'):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config.hidden_size)
            lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * config.num_layers)
            if is_training and config.keep_prob > 0:
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=config.keep_prob)

        with tf.variable_scope('backward_multi'):
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config.hidden_size)
            lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * config.num_layers)
            if is_training and config.keep_prob > 0:
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=config.keep_prob)

        if is_training and config.keep_prob > 0:
            inputs = tf.nn.dropout(inputs, config.keep_prob, noise_shape=None, seed=None)

        inputs = [tf.squeeze(x) for x in tf.split(0, config.batch_size, inputs)]

        self._output, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell,
                                                    lstm_bw_cell,
                                                    inputs,
                                                    dtype=tf.float32)

    @property
    def cells(self):
        return self._cells

    @property
    def output(self):
        return self._output


class slot1_entity_attribute(object):
    def __init__(self, is_training, config, rnn_input):
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        # target for cross entropy
        self._targets = tf.placeholder(tf.int32, [config.batch_size, config.slot1_classes], name="slot1_targets")

        rnn_tensor = tf.reduce_mean(rnn_input, 1)
        with tf.variable_scope("softmax_slot1", reuse=None, initializer=initializer):
            softmax_w = tf.get_variable("softmax_w", [config.hidden_size * 2, config.slot1_classes], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [config.slot1_classes], dtype=data_type())
            self._logits = tf.matmul(rnn_tensor, softmax_w) + softmax_b

            if config.get_summary:
                variable_summaries(softmax_w, "linear_classifier_w")
                variable_summaries(softmax_b, "linear_classifier_b")
            # weighted_cross_entropy_with_logits
            loss = tf.nn.weighted_cross_entropy_with_logits(self._logits,
                                                            tf.to_float(self._targets), 0.8)

            self._loss = tf.reduce_mean(loss)

            if config.get_summary:
                tf.scalar_summary('cross entropy', self._loss)
            self._cost = cost = tf.reduce_sum(self._loss) / config.batch_size

            self._predictions = tf.nn.softmax(self.logits)

            self._init_op = tf.initialize_all_variables()

            # No need for gradients if not training
            if not is_training:
                return

            if config.get_summary:
                self._merged_summary = tf.merge_all_summaries()

            # Optimizer.
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(10.0,
                                                       global_step,
                                                       5000,
                                                       0.1,
                                                       staircase=True)
            self._lr = learning_rate
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
            self._optimizer = optimizer.apply_gradients(zip(gradients, v),
                                                        global_step=global_step)

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

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
    def loss(self):
        return self._loss


class slot3_polarity(object):
    def __init__(self, is_training, config, rnn_output, embedding):

        rnn_tensor = tf.reduce_mean(rnn_output, 2)

        self._targets = tf.placeholder(tf.int32, [config.batch_size, config.slot3_classes], name="slot3_targets")  # 20 x 13

        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        # Apply attention
        with tf.variable_scope("attention", initializer=initializer, reuse=None):

            w_et = tf.get_variable("w_et", [config.batch_size, config.embedding_size], dtype=data_type())
            w_at = tf.get_variable("w_at", [config.batch_size, config.embedding_size], dtype=data_type())

            self._input_e = tf.placeholder(tf.int32, [config.batch_size])
            self._input_a = tf.placeholder(tf.int32, [config.batch_size])

            v_entities = tf.nn.embedding_lookup(embedding, self._input_e)
            v_aspects = tf.nn.embedding_lookup(embedding, self._input_a)

            e_t = tf.matmul(tf.matmul(tf.squeeze(v_entities), tf.transpose(w_et)), rnn_tensor)
            a_t = tf.matmul(tf.matmul(tf.squeeze(v_aspects), tf.transpose(w_at)), rnn_tensor)

            self.entity_attention = tf.nn.softmax(e_t, name="entity_attention")
            self.aspects_attention = tf.nn.softmax(a_t, name="aspect_attention")

            r_attention = tf.concat(1, [tf.matmul(self.entity_attention, rnn_tensor, transpose_b=True),
                                        tf.matmul(self.aspects_attention, rnn_tensor, transpose_b=True)])

        with tf.variable_scope("softmax_slot3", reuse=None, initializer=initializer):
            softmax_w = tf.get_variable("softmax_w", [config.batch_size * 2, config.slot3_classes], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [config.slot3_classes], dtype=data_type())
            self._logits = tf.matmul(r_attention, softmax_w) + softmax_b

            if config.get_summary:
                variable_summaries(softmax_w, "linear_classifier_w")
                variable_summaries(softmax_b, "linear_classifier_b")

            # target is valid distribution, is one hot ok on multi-class?
            loss = tf.nn.softmax_cross_entropy_with_logits(self._logits,
                                                           tf.to_float(self._targets))

            self._loss = tf.reduce_mean(loss)
            if config.get_summary:
                tf.scalar_summary('cross entropy', self._loss)
            self._cost = cost = tf.reduce_sum(self._loss) / config.batch_size

            self._predictions = tf.nn.softmax(self.logits)

            self._init_op = tf.initialize_all_variables()

            if config.get_summary:
                self._merged_summary = tf.merge_all_summaries()

            # No need for gradients if not training
            if not is_training:
                return

            # Optimizer.
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(10.0,
                                                       global_step,
                                                       5000,
                                                       0.1,
                                                       staircase=True)
            self._lr = learning_rate
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self._optimizer = optimizer.apply_gradients(zip(gradients, v),
                                                        global_step=global_step)

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

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
    def loss(self):
        return self._loss

    @property
    def entity_input(self):
        return self._input_e

    @property
    def attribute_input(self):
        return self._input_a


class multitask(object):
    """Mult-task config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 128
    # max_epoch = 25
    max_max_epoch = 1
    keep_prob = 0.80
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
    slot1_classes = 13
    slot3_classes = 4
    input_size = 100
    embedding_size = 100
    max_sequence = 70
    get_summary = False


def run_epoch(session, m, x_data, y_data, y_polarity, writer=None, run_options=None, run_metadata=None, verbose=False,
              category=False, config=False):
    epoch_size = ((len(x_data) // config.batch_size) - 1) // config.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0

    #state = m.slot1_model.initial_state.eval()

    slot1_losses = []
    slot3_losses = []

    for step, (x, y_1, y_3, e, a) in enumerate(semeval_itterator(x_data,
                                                                 y_data,
                                                                 config.batch_size,
                                                                 config.num_steps,
                                                                 category=category,
                                                                 polarity=y_polarity)):
        if writer:

            slot1_cost, slot3_cost, \
            slot1_loss, slot3_loss, \
            summary, _, _ = session.run([m.slot1_model.cost, m.slot3_model.cost,
                                         m.slot1_model.loss, m.slot3_model.loss,
                                         m.merged_summary,
                                        m.slot1_model.optimizer, m.slot3_model.optimizer],
                                       {m.embeddings.inputs: x,
                                        m.slot1_model.targets: y_1, m.slot3_model.targets: y_3,
                                        m.slot3_model.entity_input: e, m.slot3_model.attribute_input: a},
                                                           options=run_options,
                                                           run_metadata=run_metadata)
        else:
            slot1_cost, slot3_cost, \
            slot1_loss, slot3_loss, _, _ = session.run([m.slot1_model.cost, m.slot3_model.cost,
                                                     m.slot1_model.loss, m.slot3_model.loss,
                                                     m.slot1_model.optimizer, m.slot3_model.optimizer],
                                                    {m.embeddings.source_data: x,
                                                     m.slot3_model.entity_input: e, m.slot3_model.attribute_input: a,
                                                     m.slot1_model.targets: y_1, m.slot3_model.targets: y_3})

        if writer:
            writer.add_summary(summary, step)

        costs += slot1_cost
        iters += config.num_steps
        slot1_losses.append(slot1_loss)
        slot3_losses.append(slot3_loss)

        if verbose and iters % (config.batch_size * 5) == 0:
            print("step %.3f slot_loss : %.6f slot3_loss: %.6f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, slot1_loss, slot3_loss, iters * config.batch_size / (time.time() - start_time)))

    return np.exp(abs(costs) / iters), slot1_losses, slot3_losses


class slot1_and_slot3_model(object):
    def __init__(self, is_training, config):
        if config.get_summary:
            self._merged_summary = tf.merge_all_summaries()
        self._embeddings = shared_Embeddings(config)
        self._rnn_output = shared_BiRNN(is_training,
                                        config,
                                        inputs=self._embeddings.inputs)

        """
            Both Slot1 and Slot3 Models share BiRNN above Representations
        """

        # Slot1 Model
        self._slot1_model = slot1_entity_attribute(is_training,
                                                   config,
                                                   self._rnn_output.output)

        # Slot 3 Model
        self._slot3_model = slot3_polarity(is_training,
                                           config,
                                           self._rnn_output.output,
                                           self._embeddings.embeddings)

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def birrn_output(self):
        return self._rnn_output

    @property
    def slot1_model(self):
        return self._slot1_model

    @property
    def slot3_model(self):
        return self._slot3_model

    @property
    def merged_summary(self):
        return self._merged_summary


def train_task(CONST, data):
    config = multitask()

    config.vocab_size = len(data["embeddings"])

    tf.reset_default_graph()

    with tf.Graph().as_default(), tf.Session() as session:

        with tf.variable_scope("model", reuse=None, initializer=tf.contrib.layers.xavier_initializer()):
            training_model = slot1_and_slot3_model(is_training=True, config=config)  # model class

        if CONST.TRAIN:
            tf.initialize_all_variables().run()

            # Check if Model is Saved and then Load
            if check_saved_file(CONST.MULTITASK_CHECKPOINT_PATH + "checkpoint"):
                saver = tf.train.Saver()
                saver.restore(sess=session, save_path=CONST.MULTITASK_CHECKPOINT_PATH)

            session.run(training_model.embeddings.embeddings.assign(data["embeddings"]))  # train

            slot1_losses = []
            slot3_losses = []

            train_writer = run_metadata = run_options = False

            if config.get_summary:
                # session = tf.InteractiveSession()
                train_writer = tf.train.SummaryWriter(
                    CONST.SLOT1_MODEL_PATH + "attention_graph/" + config.__class__.__name__,
                    session.graph)

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            for i in range(config.max_max_epoch):  # setup epoch

                train_perplexity, slot1_loss, slot3_loss = run_epoch(session,
                                                                     training_model,
                                                                     data["x_train"],
                                                                     data["y_train"],
                                                                     data["p_train"],
                                                                     train_writer,
                                                                     run_options,
                                                                     run_metadata,
                                                                     category=data["a_train"],
                                                                     verbose=True,
                                                                     config=config)

                slot1_losses = slot1_losses + [np.mean(slot1_loss)]
                slot3_losses = slot3_losses + [np.mean(slot3_loss)]

                print("Epoch: %d Avg. Total Mean Loss slot1: %.6f slot3: %.6f" % (i + 1,
                                                                               np.mean(slot1_losses),
                                                                               np.mean(slot3_losses)))

            # Output Config/Losses
            from util.evaluations import print_config
            print_config(config)

            # Save CheckPoint
            saver = tf.train.Saver(tf.all_variables())
            path = saver.save(sess=session, save_path=CONST.MULTITASK_CHECKPOINT_PATH)
            print("model saved: " + path)

            # Save Losses for Later
            loss = {"slot1": slot1_losses, "slot3": slot3_losses}
            save_pickle(CONST.DATA_DIR + config.__class__.__name__ + "multi_task", loss)

            # Try and plot the losses
            import matplotlib.pyplot as plt
            x = [i for i in range(len(slot1_losses))]
            plt.plot(np.array(x), np.array(slot1_losses))
            plt.plot(np.array(x), np.array(slot3_losses))
            plt.savefig("losses_" + config.__class__.__name__ + ".png")

        if not CONST.TRAIN:
            with tf.variable_scope("model", reuse=True):  # reuse scope to evaluate model
                validation_model = slot1_and_slot3_model(is_training=True, config=config)  # model class

                session.run(validation_model.embeddings.embeddings.assign(data["embeddings"]))  # load embeddings

                saver = tf.train.Saver()
                saver.restore(sess=session, save_path=CONST.MULTITASK_CHECKPOINT_PATH)

                slot1_predictions = []
                slot3_predictions = []

                # Get Predictions
                for step, (x, y_1, y_3, e, a) in enumerate(semeval_itterator(data["x_test"],
                                                                             data["y_test"],
                                                                             config.batch_size,
                                                                             config.num_steps,
                                                                             category=data["aspects"],
                                                                             polarity=data["p_test"])):

                    slot1_prediction, slot3_prediction = session.run([validation_model.slot1_model.predictions,
                                                                      validation_model.slot3_model.predicitions],
                                                                     {validation_model.input_data: x,
                                                                      validation_model.targets: y,
                                                                      validation_model.input_e: e,
                                                                      validation_model.input_a: a})

                    slot1_predictions = slot1_predictions + slot1_prediction.tolist()
                    slot3_predictions = slot3_predictions + slot3_prediction.tolist()

                even_batch = len(data["x_test"]) % config.batch_size
                remove_added_batch = config.batch_size - even_batch

                del slot1_predictions[-remove_added_batch:]
                del slot3_predictions[-remove_added_batch:]

                slot1_predictions = np.asarray(slot1_predictions)
                slot3_predictions = np.asarray(slot3_predictions)

                # print congiuration for test predictions
                from util.evaluations import print_config
                print_config(config)

                # save predictions
                predictions = {"slot1": slot1_predictions, "slot3": slot3_predictions,
                               "slot1_y": data["y_test"], "slot3_y": data["p_test"]}
                save_pickle(CONST.DATA_DIR + config.__class__.__name__ + "_predictions",
                            predictions)
                print("predictions saved to file ", CONST.DATA_DIR + config.__class__.__name__ + "_predictions")

                from util.evaluations import evaluate_multilabel
                from util.evaluations import evaluate_multiclass
                evaluate_multilabel(predictions.slot1, predictions.slot1_y, CONST.THRESHOLD)
                evaluate_multiclass(predictions.slot3, predictions.slot3_y, True)


if __name__ == "__main__":
    # Set and Overload Arguments
    CONST.parse_argument(argparse.ArgumentParser())

    # Set Time of Experiment
    now = datetime.datetime.now()
    time_stamp = "_".join([str(a) for a in [now.month, now.day, now.hour, now.minute, now.second]])

    data = load(CONST.DATA_DIR + CONST.DATA_FILE)

    if CONST.TRAIN:
        train_task(CONST, data)
