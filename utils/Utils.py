# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image
import shutil


import json
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def load_data():
    graphName = FLAGS.network_node
    graphPath = "Data_Set/" + FLAGS.dataset_name + "/"  + str(graphName) + "-node/"

    prefix = graphPath + str(graphName)
    Data_training = json.load(open(prefix + "-node-training.json"))
    Data_testing = json.load(open(prefix + "-node-testing.json"))
    return Data_training, Data_testing

def load_Newdata():
    graphPath = "Data_Set/" + FLAGS.dataset_name + "/"
    prefix = graphPath
    Data_training = json.load(open(prefix + "training.json"))
    Data_testing = json.load(open(prefix + "testing.json"))
    return Data_training, Data_testing

def load_data_new_y():
    generatedGraphsPath = "Data_Set/generatedGraphsData/"
    graphName = FLAGS.network_node
    graphPath = generatedGraphsPath + str(graphName) + "-node/"

    prefix = graphPath + str(graphName)
    Data_y_new_target = json.load(open(prefix + "-node-Y_new_target.json"))
    return Data_y_new_target

def load_data_new_tp():
    graphName = FLAGS.network_node
    graphPath = "Data_Set/" + FLAGS.dataset_name + "/" + str(graphName) + "-node-valid/"

    prefix = graphPath + str(graphName)
    Data_training = json.load(open(prefix + "-node-training.json"))
    Data_testing = json.load(open(prefix + "-node-testing.json"))
    return Data_training, Data_testing

