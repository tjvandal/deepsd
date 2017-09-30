import os
import time
import sys
import pickle
import datetime as dt
import re

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

base_srcnn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'srcnn-tensorflow')
sys.path.append(base_srcnn)
from srcnn import srcnn
from tfreader import inputs_climate

flags = tf.flags

# model hyperparamters
flags.DEFINE_string('hidden', '64,32,1', 'Number of units in hidden layer 1.')
flags.DEFINE_string('kernels', '9,1,5', 'Kernel size of layer 1.')

# Model training parameters
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 50000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('test_batch', 10, 'Batch size.')

# what do these training images look like?
flags.DEFINE_integer('input_size', 38, 'Number of input channels.')
flags.DEFINE_integer('input_depth', 2, 'Number of input channels.')
flags.DEFINE_integer('output_depth', 1, 'Number of output channels.')

# Decay and bayesian parameters for uncertianty quantification
flags.DEFINE_float('decay', 0.0000, 'Weight decay term.')
flags.DEFINE_float('keep_prob', 1.0, 'Dropout Probability.')
flags.DEFINE_integer('mc_steps', 1, 'Number of MC steps during test time.')

# when to save, plot, and test
flags.DEFINE_integer('save_step', 1000, 'How often should I save the model')
flags.DEFINE_integer('test_step', 50, 'How often test steps are executed and printed')

# where to save things
flags.DEFINE_string('data_dir', 'scratch/ppt_008_016/', 'Data Location')
flags.DEFINE_string('save_dir', 'scratch/', 'Where to save checkpoints + logs.')
flags.DEFINE_string('transfer_learning', 'scratch/srcnn_ppt_004_008_64-32-1_9-3-5/',
                    'Load a previous checkpoint for transfer learning, starting in a good place.')

# parse flags
FLAGS = flags.FLAGS
FLAGS._parse_flags()

if FLAGS.keep_prob == 1.0:
    FLAGS.mc_steps = 1

# where to save and get data
DATA_DIR = FLAGS.data_dir
data_name = os.path.basename(DATA_DIR.strip("/"))
timestamp = str(int(time.time()))
curr_time = dt.datetime.now()

SAVE_DIR = os.path.join(FLAGS.save_dir, "srcnn_%s_%s_%s" % ( data_name,
                    FLAGS.hidden.replace(",", "-"),
                    FLAGS.kernels.replace(",", "-")))

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

# parse architecture from flags
HIDDEN_LAYERS = [int(x) for x in FLAGS.hidden.split(",")]
KERNELS = [int(x) for x in FLAGS.kernels.split(",")]

def skill_score_tf(x,y,bin_width=1.):
    xy = tf.concat(0, [tf.reshape(x, [-1]), tf.reshape(x, [-1])])
    cdfmin = tf.reduce_min(xy)
    cdfmax = tf.reduce_max(xy)
    bins = tf.to_int32(1. * (cdfmin - cdfmax) / bin_width)
    xhist = tf.histogram_fixed_width(x, [cdfmin, cdfmax], nbins=500, dtype=tf.float32) / tf.to_float(tf.size(x))
    yhist = tf.histogram_fixed_width(y, [cdfmin, cdfmax], nbins=500, dtype=tf.float32) / tf.to_float(tf.size(y))
    return tf.reduce_sum(tf.minimum(xhist, yhist))

def fill_na(x, fillval=0):
    fill = tf.ones_like(x) * fillval
    return tf.select(tf.is_finite(x), x, fill)

def mask(label, prediction):
    fill = tf.ones_like(label) * np.nan
    return tf.select(tf.is_finite(label), prediction, fill)

def train():
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        global_step = tf.get_variable('global_step', [],
                    initializer=tf.constant_initializer(0), trainable=False)

        errors = []

        # lets get data to iterate through
        hr_size = FLAGS.input_size
        lr_size = hr_size / 2
        train_images, train_labels = inputs_climate(FLAGS.batch_size, FLAGS.num_epochs,
                        DATA_DIR, lr_shape=[lr_size, lr_size], lr_d=(FLAGS.input_depth-1),
                        aux_d=1, is_training=True,
                        hr_shape=[hr_size, hr_size], hr_d=FLAGS.output_depth)
        test_images, test_labels, test_times = inputs_climate(FLAGS.test_batch, FLAGS.num_epochs,
                        DATA_DIR, is_training=False, lr_d=(FLAGS.input_depth-1), aux_d=1,
                        hr_d=FLAGS.output_depth)

        # crop training labels
        border_size = (sum(KERNELS) - len(KERNELS))/2
        train_labels_cropped = train_labels[:,border_size:-border_size,border_size:-border_size,:]

        # set placeholders
        is_training = tf.placeholder_with_default(True, (), name='is_training')

        x = tf.cond(is_training, lambda: train_images, lambda: test_images)
        y = tf.cond(is_training, lambda: train_labels_cropped, lambda: test_labels)

        x = tf.identity(x, name='x')
        y = tf.identity(y, name='y')

        # Use SRCNN
        model = srcnn.SRCNN(x, y, HIDDEN_LAYERS, KERNELS, input_depth=FLAGS.input_depth,
                            learning_rate=FLAGS.learning_rate, upscale_factor=2,
                           is_training=is_training, gpu=True)
        prediction = tf.identity(model.prediction, name='prediction')

        # initialize graph and start session
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=False))
        sess.run(init)
        sess.run(tf.local_variables_initializer())

        # look for checkpoint
        try:
            checkpoint = tf.train.latest_checkpoint(FLAGS.transfer_learning)
            saver.restore(sess, checkpoint)
            print "Checkpoint", checkpoint
        except tf.errors.InternalError as err:
            print "Warning: Could not find checkpoint", err
            pass

        # start coordinator for data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # summary data
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(SAVE_DIR + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SAVE_DIR + '/test', sess.graph)

        def feed_dict(train=True):
            return {is_training: train}

        #curr_step = int(sess.run(model.global_step))
        curr_step = 0
        for step in range(curr_step, FLAGS.num_epochs+1):
            start_time = time.time()
            _, train_loss, train_rmse = sess.run([model.opt, model.loss, model.rmse],
                                                 feed_dict=feed_dict(True))
            duration = time.time() - start_time
            if step  % FLAGS.test_step == 0:
                test_summary = sess.run(summary_op, feed_dict=feed_dict(True))
                train_writer.add_summary(test_summary, step)

                d = feed_dict(train=True)
                out = sess.run([model.loss, model.rmse, summary_op, model.x_norm], feed_dict=d)
                print np.mean(out[3])
                test_writer.add_summary(out[2], step)
                print "Step: %d, Examples/sec: %0.5f, Training Loss: %2.3f," \
                        " Train RMSE: %2.3f, Test RMSE: %2.4f" % \
                        (step, FLAGS.batch_size/duration, train_loss, train_rmse, out[1])

            if step % FLAGS.save_step == 0:
                save_path = saver.save(sess, os.path.join(SAVE_DIR, "srcnn.ckpt"))
                print SAVE_DIR

        save_path = saver.save(sess, os.path.join(SAVE_DIR, "srcnn.ckpt"))

if __name__ == "__main__":
    train()
