#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2018  
#
# This file is part of Virtaal.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

""" CIFAR-10 data set.
See http://www.cs.toronto.edu/~kriz/cifar.html.
author: chenweiwei@ict.ac.cn

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import cPickle
import os 
import download

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf



def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.

    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]

    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.

    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.

    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]

#directory to download and store the dataet
data_path = "./"
#directory after extract the cifar-10-python.tar.gz
data_dir = "cifar-10-batches-py/"
#URL for the cifar10 on the internet
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

#width and height for each image
img_size =32

#number of channels for each image, 3 channel for red, green, blue
num_channels =3

#length  of an image when flattened into a 1-dim array
img_size_flat = img_size*img_size*num_channels

#number of classes
num_classes = 10 

#num of  images for the training set, 50000 for cifar10 in the training
_num_imges_train = 50000  

def _get_file_path(filename = ''):
    """
    return the full path of the data-file
    if filename=='' then return the directory of the files
    """
    return os.path.join(data_path,"cifar-10-batches-py/",filename)


def _unpickle(filename):
    """
    unpickle the given file and return the data
    """
    file_path = _get_file_path(filename)
    print("loading the data from:", file_path)

    with open(file_path,mode="rb") as file:
        data = cPickle.load(file)
    return data

def _covert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """
    raw_float = np.array(raw,dtype=float)/255.0
    images = raw_float.reshape([-1,num_channels,img_size,img_size])
    images = images.transpose([0,2,3,1])

    return images

def _load_data(filename):
    data = _unpickle(filename)
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])
    images = _covert_images(raw_images)

    return images, cls


def maybe_download_and_extract():
    """
    Download and extract the CIFAR-10 data-set if it doesn't already exist
    in data_path (set this variable first to the desired path).
    """
    download.maybe_download_and_extract(download_dir=data_path)


def load_class_names():
    """
    Load the names for the classes in the CIFAR-10 data-set.

    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """

    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _get_file_names():
    """Returns the file names expected to exist in the input_dir."""
    file_names = {}
    file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 5)]
    file_names['validation'] = ['data_batch_5']
    file_names['eval'] = ['test_batch']
    return file_names

def read_pickle_from_file(filename):
    with tf.gfile.Open(filename, 'r') as f:
        data_dict = cPickle.load(f)
    return data_dict


def convert_to_tfrecord(input_files, output_file):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = read_pickle_from_file(input_file)
            data = data_dict['data']
            labels = data_dict['labels']
            num_entries_in_batch = len(labels)
            for i in range(num_entries_in_batch):
                example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': _bytes_feature(data[i].tobytes()),
                    'label': _int64_feature(labels[i])
                }))
                record_writer.write(example.SerializeToString())

def download_cifar10_convert_to_tfrecord():
    print("Download from {} and extract.".format(data_url))
    maybe_download_and_extract()
    filenames = _get_file_names()
    input_dir = os.path.join(data_path,data_dir)
    for mode, files in filenames.items():
        input_files = [os.path.join(input_dir,x) for x in files]
        output_file = os.path.join(data_path, mode+'.tfrecords')
        try:
            os.remove(output_file)
        except OSError:
            pass
        convert_to_tfrecord(input_files,output_file)
    print("Done!")


if __name__ == '__main__':
    download_cifar10_convert_to_tfrecord()
