#import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import sys
import os
import time
import tensorflow as tf
import pickle

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def convert_to_tf(inputs, elevs, labels, lats, lons, t, filename):
    #inputs, labels = data_transformer.transform(inputs, labels)
    writer = tf.python_io.TFRecordWriter(filename)
    num_examples, height, width, depth = labels.shape
    label_depth = labels.shape[3]
    for index in range(num_examples):
        img_in = inputs[index].astype(np.float32).tostring()
        elev_in = elevs[index].astype(np.float32).tostring()
        img_lab = labels[index].astype(np.float32).tostring()
        lat_in = lats[index].astype(np.float32).tostring()
        lon_in = lons[index].astype(np.float32).tostring()
        time_in = t[index].astype(np.int)
        example = tf.train.Example(features=tf.train.Features(feature={
            'rows': _int64_feature(height),
            'cols': _int64_feature(width),
            'input_depth': _int64_feature(depth),
            'label_depth': _int64_feature(label_depth),
            'label': _bytes_feature(img_lab),
            'img_in': _bytes_feature(img_in),
            'elev': _bytes_feature(elev_in),
            'lat': _bytes_feature(lat_in),
            'lon': _bytes_feature(lon_in),
            'time': _int64_feature(time_in),
            }))
        writer.write(example.SerializeToString())
    writer.close()
