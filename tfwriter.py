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
    n, hr_h, hr_w, hr_d = labels.shape
    _, lr_h, lr_w, lr_d = inputs.shape
    assert elevs.shape[1] == hr_h
    for index in range(n):
        img_in = inputs[index].astype(np.float32).tostring()
        elev_in = elevs[index].astype(np.float32).tostring()
        img_lab = labels[index].astype(np.float32).tostring()
        lat_in = lats[index].astype(np.float32).tostring()
        lon_in = lons[index].astype(np.float32).tostring()
        time_in = t[index].astype(np.int)
        example = tf.train.Example(features=tf.train.Features(feature={
            'hr_h': _int64_feature(hr_h),
            'hr_w': _int64_feature(hr_w),
            'hr_d': _int64_feature(hr_d),
            'lr_h': _int64_feature(lr_h),
            'lr_w': _int64_feature(lr_w),
            'lr_d': _int64_feature(lr_d),
            'label': _bytes_feature(img_lab),
            'img_in': _bytes_feature(img_in),
            'aux': _bytes_feature(elev_in),
            'lat': _bytes_feature(lat_in),
            'lon': _bytes_feature(lon_in),
            'time': _int64_feature(time_in),
            }))
        writer.write(example.SerializeToString())
    writer.close()
