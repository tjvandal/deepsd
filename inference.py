import os, sys
import tensorflow as tf
import ConfigParser
from tensorflow.python.framework import graph_util
import numpy as np
import xarray as xr
import cv2
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(basedir, 'datasets'))
import prism
import utils

# model parameters
flags = tf.flags

FLAGS = flags.FLAGS
FLAGS._parse_flags()

def get_graph_def():
    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        new_saver = tf.train.import_meta_graph(checkpoint + '.meta')
        new_saver.restore(sess, checkpoint)
        return sess.graph_def

def freeze_graph(model_folder, graph_name=None):
    # We start a session and restore the graph weights
    with tf.Session() as sess:
        # We retrieve our checkpoint fullpath
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path

        # We precise the file fullname of our freezed graph
        absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
        output_graph = absolute_model_folder + "/frozen_model.pb"
        if os.path.exists(output_graph):
            os.remove(output_graph)

        # Before exporting our graph, we need to precise what is our output node
        # This is how TF decides what part of the Graph he has to keep and what part it can dump
        # NOTE: this variable is plural, because you can have multiple output nodes
        if graph_name is not None:
            output_node_names = "prediction"
        else:
            raise ValueError("Give me a graph_name")

        # We clear devices to allow TensorFlow to control on which device it will load operations
        clear_devices = True 

        # We import the meta graph and retrieve a Saver
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                           clear_devices=clear_devices)

        # We retrieve the protobuf graph definition
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        black_list = []
        saver.restore(sess, input_checkpoint)

        # Retrieve the protobuf graph definition and fix the batch norm nodes
        gd = sess.graph.as_graph_def()
        for node in gd.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            gd, # The graph_def is used to retrieve the nodes 
            output_node_names.split(","), # The output node names are used to select the usefull nodes
            variable_names_blacklist=black_list
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

def load_graph(frozen_graph_filename, graph_name, x=None):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    #with tf.Graph().as_default() as graph:
        is_training = tf.constant(False)
        if x is None:
            x = tf.placeholder(tf.float32, shape=(None, None, None, 2), name="%s/new_x" % graph_name)
        y, = tf.import_graph_def(
            graph_def,
            input_map={'x': x, 'is_training': is_training},
            return_elements=['prediction:0'],
            name=graph_name,
            op_dict=None,
            producer_op_list=None
        )
    return y #graph, y

def join_graphs(checkpoints, new_checkpoint):
    '''
    placeholders:
        low-resolution ppt
        elevation for each checkpoint

    x = concat([ppt, elev_1])
    for each checkpoint:
        x -> y
        x = concat([y, elev_i])
    return y
    '''
    # begin by freezing each graph independently
    for cpt in checkpoints:
        # freeze current graph
        graph_name = "_".join(os.path.basename(cpt.strip("/")).split("_")[1:4])
        freeze_graph(cpt, graph_name)
        tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=(None, None, None, 1), name="lr_x")
    elevs = []
    for j, cpt in enumerate(checkpoints):
        # another elevation placeholder
        elv = tf.placeholder(tf.float32, shape=(None, None, None, 1), name="elev_%i" % j)
        elevs.append(elv)

        # resize low-resolution
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        size = tf.stack([h*2, w*2])
        x = tf.image.resize_bilinear(x, size)

        # join elevation and interpolated image
        x = tf.concat([x, elv], axis=3)
        graph_name = "_".join(os.path.basename(cpt.strip("/")).split("_")[1:4])

        # load frozen graph with x as the input
        print 'x', x
        next_input = graph_name + '/x'
        x = load_graph(os.path.join(cpt, 'frozen_model.pb'), graph_name, x=x)

        for var in tf.global_variables():
            print 'chkp', j, var.op.name
            time.sleep(0.1)

    with tf.Session() as sess:
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(new_checkpoint, sess.graph)
        train_writer.add_graph(tf.get_default_graph())

        gd = sess.graph.as_graph_def()
        output_graph = os.path.join(new_checkpoint, 'frozen_graph.pb')
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(gd.SerializeToString())
        print("%d ops in the final graph." % len(gd.node))

    tf.reset_default_graph()
    return output_graph

def main(frozen_graph, scale1=1., scale2=1./2, n_stacked=1, upscale_factor=2):
    # read configuration file
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
    config = ConfigParser.ConfigParser()
    config.read(config_file)

    year = 2015
    # read prism dataset
    ## resnet parameter will not re-interpolate X
    dataset = prism.PrismSuperRes(os.path.join(config.get('Paths', 'prism'), 'ppt','raw'), year,
                      config.get('Paths', 'elevation'), model='srcnn')

    X, elev, Y, lats, lons, times = dataset.make_test(scale1=scale1, scale2=scale2)
    print X.shape, elev.shape, Y.shape
    mask = (Y[0,:,:,0]+1)/(Y[0,:,:,0] + 1)
    elev_hr = elev[0,:,:,0] # all the elevations are the same, remove some data from memory

    #  resize x
    n, h, w, c = X.shape
    print 'Y Shape', Y.shape
    print 'X shape', X.shape

    # get elevations at all 5 resolutions
    elev_dict = {}
    elevs = []
    for i in range(n_stacked):
        r = upscale_factor**i
        elev_dict[1./r] = cv2.resize(elev_hr, (0,0), fx=1./r, fy=1./r)
        print 'elev shape', elev_dict[1./r].shape
        elevs.append(tf.constant(elev_dict[1./r][np.newaxis, :, :, np.newaxis].astype(np.float32)))
    elevs = elevs[::-1]

    #now read in frozen graph, set placeholder for x, constant for elevs
    x = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    #elev_1  = tf.constant(elev_dict[1.][np.newaxis, :, :, np.newaxis].astype(np.float32))
    #elev_0  = tf.constant(elev_dict[1./2][np.newaxis, :, :, np.newaxis].astype(np.float32))
    input_map= {'elev_%i' % i: elevs[i] for i in range(n_stacked)}
    input_map['lr_x'] = x

    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        y, = tf.import_graph_def(
            graph_def,
            input_map=input_map,
            return_elements=['ppt_008_016/prediction:0'],
            name='deepsd',
            op_dict=None,
            producer_op_list=None
        )

    downscaled = []
    with tf.Session() as sess:
        rmses = []
        for i in range(0,len(X)):
            _x = X[i,np.newaxis]
            # is_training=False removes padding at test time
            downscaled += [sess.run(y,feed_dict={x: _x})]
            rmses.append(np.sqrt(np.nanmean((downscaled[-1] - Y[i])**2)))
        print "RMSE", np.mean(rmses)

    downscaled = np.concatenate(downscaled, axis=0)
    downscaled *= mask[:,:,np.newaxis]
    precip = xr.DataArray(downscaled[:,:,:,0], coords=[times, lats[0], lons[0]], 
                          dims=['time', 'lat', 'lon'])
    xr.Dataset({'precip': precip}).to_netcdf("precip_%i_downscaled.nc" % year)

    fig, axs = plt.subplots(3,1)
    axs = np.ravel(axs)
    axs[0].imshow(Y[0,:,:,0], vmax=ymax)
    axs[0].axis('off')
    axs[0].set_title("Observed")
    axs[1].imshow(_x[0,:,:,0])
    axs[1].axis('off')
    axs[1].set_title("Input")
    axs[2].imshow(downscaled[0,:,:,0] * mask, vmax=ymax)
    axs[2].axis('off')
    axs[2].set_title("Downscaled")
    plt.savefig('res.pdf')
    plt.close()

if __name__ == '__main__':
    checkpoints = [
        'scratch/srcnn_ppt_008_016_64-32-1_9-1-5/',
    ]
    checkpoints = sorted(checkpoints)[::-1]
    joined_checkpoint = os.path.join(os.path.dirname(checkpoints[0][:-1]), 'joined_008_016')
    if not os.path.exists(joined_checkpoint):
        os.mkdir(joined_checkpoint)

    #new_graph = join_graphs(checkpoints, joined_checkpoint)
    new_graph = os.path.join(joined_checkpoint, 'frozen_graph.pb')
    main(new_graph, scale1=1./2, scale2=1./2)
