import os
import time
import sys
import datetime as dt
import re

import numpy as np
import tensorflow as tf
import ConfigParser

base_srcnn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'srcnn-tensorflow')
sys.path.append(base_srcnn)
from srcnn import srcnn
from tfreader import inputs_climate


flags = tf.flags
flags.DEFINE_string('config_file', 'config.ini', 'Configuration file with [SRCNN] section.')
flags.DEFINE_string('checkpoint_file', None, 'Any checkpoint with the same architecture as'\
                    'configured.')
flags.DEFINE_string('experiment_number', '1', 'Experiment-? in config file/')

# parse flags
FLAGS = flags.FLAGS
FLAGS._parse_flags()



## READ CONFIGURATION FILE
config = ConfigParser.ConfigParser()
config.read(FLAGS.config_file)

LAYER_SIZES = [int(k) for k in config.get('SRCNN', 'layer_sizes').split(",")]
KERNEL_SIZES = [int(k) for k in config.get('SRCNN', 'kernel_sizes').split(",")]
OUTPUT_DEPTH = LAYER_SIZES[-1]
AUX_DEPTH = int(config.get('SRCNN', 'aux_depth'))
LEARNING_RATE = float(config.get('SRCNN', 'learning_rate'))
TRAINING_ITERS = int(config.get('SRCNN', 'training_iters'))
BATCH_SIZE = int(config.get('SRCNN', 'batch_size'))
TRAINING_INPUT_SIZE = int(config.get('SRCNN', 'training_input_size'))
INPUT_DEPTH = int(config.get('SRCNN', 'training_input_depth'))
SAVE_STEP = int(config.get('SRCNN', 'save_step'))
TEST_STEP = int(config.get('SRCNN', 'test_step'))
KEEP_PROB = 1. - float(config.get('SRCNN', 'dropout_prob'))

# where to save and get data
DATA_DIR = config.get('Experiment-%s' % FLAGS.experiment_number, 'data_dir')
data_name = os.path.basename(DATA_DIR.strip("/"))
timestamp = str(int(time.time()))
curr_time = dt.datetime.now()

SAVE_DIR = os.path.join(config.get('SRCNN', 'scratch'), "srcnn_%s_%s_%s" % ( data_name,
                    '-'.join([str(s) for s in LAYER_SIZES]),
                    '-'.join([str(s) for s in KERNEL_SIZES])))

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

def train():
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        global_step = tf.get_variable('global_step', [],
                    initializer=tf.constant_initializer(0), trainable=False)

        errors = []

        # lets get data to iterate through
        lr_size = int(TRAINING_INPUT_SIZE / 2)
        train_images, train_labels = inputs_climate(BATCH_SIZE, TRAINING_ITERS,
                        DATA_DIR, lr_shape=[lr_size, lr_size], lr_d=INPUT_DEPTH,
                        aux_d=AUX_DEPTH, is_training=True,
                        hr_shape=[TRAINING_INPUT_SIZE, TRAINING_INPUT_SIZE], hr_d=OUTPUT_DEPTH)
        test_images, test_labels, test_times = inputs_climate(BATCH_SIZE, TRAINING_ITERS,
                        DATA_DIR, is_training=False, lr_d=INPUT_DEPTH, aux_d=1,
                        hr_d=OUTPUT_DEPTH)

        # crop training labels
        border_size = (sum(KERNEL_SIZES) - len(KERNEL_SIZES))/2
        train_labels_cropped = train_labels[:,border_size:-border_size,border_size:-border_size,:]

        # set placeholders
        is_training = tf.placeholder_with_default(True, (), name='is_training')

        x = tf.cond(is_training, lambda: train_images, lambda: test_images)
        y = tf.cond(is_training, lambda: train_labels_cropped, lambda: test_labels)

        x = tf.identity(x, name='x')
        y = tf.identity(y, name='y')

        # Use SRCNN
        model = srcnn.SRCNN(x, y, LAYER_SIZES, KERNEL_SIZES, input_depth=INPUT_DEPTH,
                            learning_rate=LEARNING_RATE, upscale_factor=2,
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
        if FLAGS.checkpoint_file is not None:
            try:
                checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_file)
                saver.restore(sess, checkpoint)
                print("Checkpoint", checkpoint)
            except tf.errors.InternalError as err:
                print("Warning: Could not find checkpoint", err)
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
        for step in range(curr_step, TRAINING_ITERS+1):
            start_time = time.time()
            _, train_loss, train_rmse = sess.run([model.opt, model.loss, model.rmse],
                                                 feed_dict=feed_dict(True))
            duration = time.time() - start_time
            if step  % TEST_STEP == 0:
                test_summary = sess.run(summary_op, feed_dict=feed_dict(True))
                train_writer.add_summary(test_summary, step)

                d = feed_dict(train=True)
                out = sess.run([model.loss, model.rmse, summary_op, model.x_norm], feed_dict=d)
                test_writer.add_summary(out[2], step)
                print("Step: %d, Examples/sec: %0.5f, Training Loss: %2.3f," \
                        " Train RMSE: %2.3f, Test RMSE: %2.4f" % \
                        (step, BATCH_SIZE/duration, train_loss, train_rmse, out[1]))

            if step % SAVE_STEP == 0:
                save_path = saver.save(sess, os.path.join(SAVE_DIR, "srcnn.ckpt"))

        save_path = saver.save(sess, os.path.join(SAVE_DIR, "srcnn.ckpt"))

if __name__ == "__main__":
    train()
