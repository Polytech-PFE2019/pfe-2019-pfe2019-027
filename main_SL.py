#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:53:17 2019

@author: raparicio
"""

# -*- coding: utf-8 -*-


#from Neural_Networks.GSMN import GSMN
from Neural_Networks.NANE_LinkToPath import NANE_LinkToPath
from Neural_Networks.SL_FC_LinkToPath import SL_FC_LinkToPath
from utils.proc import log_dir, accuracyLinkToPath
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from keras import backend as K

# change : topology, dataset, trafic generator opening of the file

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS
import json

############# Hyperparameters for neural network
flags.DEFINE_string('which_nn_use', 'NANE_LinkToPath',
                    'with neural network use :   ' ''' SL_FC_LinkToPath,GSMN,benchmarkLinkToPath ''')

#flags.DEFINE_integer('max_total_steps', 50000, 'Maximum total number of iterations')  # 1000000
flags.DEFINE_integer('epochs', 0, 'number of epochs to train.')
flags.DEFINE_integer('batch_size', 256, 'batch size')

flags.DEFINE_integer('nb_layers', 5, 'the number of hidden layers.')
flags.DEFINE_integer('size_layer', 60, 'the number of neurons in each layer (for FC) ')
flags.DEFINE_string('activation', 'relu', 'activation function for hidden layer.')
flags.DEFINE_float('learning_rate', 0.0004, 'initial learning rate.')
flags.DEFINE_float('weight_decay', 0.00003, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_boolean('batch_normalization', True, 'whether use batch_normalization.')

flags.DEFINE_string('modele_name', "foo", 'modele file name')
flags.DEFINE_string('modele_short_name', "foo", 'modele short file name')

### paras in NANE: neigh aggreg-based node embedding
#flags.DEFINE_string('which_GS_use', 'LinearOperation',
#                    'Matrixproduct.')  # using the matrix product or direct big weight matrix
flags.DEFINE_string('aggregator', 'mean', 'aggregator function: mean , mean_out_in, gcn ,adj_masked_dense ')
flags.DEFINE_integer('final_embedding_size', 6, 'Size of link embedding')
flags.DEFINE_integer('hidden_embedding_size', 6, 'Size of link embedding in hidden layer')
flags.DEFINE_integer('nb_nane_layers', 4, 'number of NANE hidden layers')
flags.DEFINE_integer('nb_fc_layers', 1, 'number of FC hidden layers ')

############# Tensorboard parameters
flags.DEFINE_boolean('show_metadata', True, ' show in tensorboard the usage of memory and compute time')
flags.DEFINE_boolean('print_data', True, ' show in tensorboard the label and the output associated')
flags.DEFINE_integer('save_every', 1, "How often to save training info.")

############# Dataset parameters
#flags.DEFINE_integer('network_node', 10, 'Can be 10, 50')
#flags.DEFINE_integer('degree', 2, ' percentage of congestion case in the data')
flags.DEFINE_integer('state', 1, 'input data to the neural networks'
                                 '''
                                 0 is edge embedding state: L vectors + one N hot-vector,
                                     where each link vector = [w_0 ... w_W vol]_l,
                                     where the N-hot vector is: vector(src) = - Vol; vector(dst) = + Vol
                                 1 is node embedding state: N vectors, each node vector = [lw_0 ...lw_LW vol]_n,
                                     where vol > 0, if node = src; vol < 0, if node = dst

                                 '''
                     )
flags.DEFINE_boolean('is_shuffling', True, 'training set is shuffle at each epoch')
flags.DEFINE_string('dataset_path', 'Data_Set/data/', 'PathToTheDataSet ')

### left to default values in main experiments
flags.DEFINE_integer('link_capacity', 1, 'maximum capacity on each link.')
flags.DEFINE_integer('numWavelengths', 4, 'numWavelengthson each link.')

### generating the traffic and data for a specific network
#flags.DEFINE_integer('Total_R', 10000, 'total number of request Generation.')
flags.DEFINE_integer('k_path_generated', 3, ' K shortes paths generated for the data set "LinkToPath" ')


### logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', 'Data_Set/Results/', 'base directory for logging and saving embeddings')
#flags.DEFINE_integer('validate_iter', 5000, 'how often to run a validation minibatch.')
#flags.DEFINE_integer('validate_batch_size', 256, 'how many nodes per validation sample.')
#flags.DEFINE_integer('network_link', 12, 'Can be 6, 12, 28 and 176')
#flags.DEFINE_integer('low_link_weight', FLAGS.network_link, 'the low link weight')
#flags.DEFINE_integer('high_link_weight', 2 * FLAGS.network_link, 'the high link weight')

def main():
    nodes = [10]
    degree = [5]
    netw_i = 0
    lpMat_i = 0
    maxUtilPerNodePair = [0.3]
#    trace_id = 0
    nbNetwork = 10
    traffic="uniform"

    topo_pathname = str(nodes[0]) + "node/" + str(degree[0]) + "degree/" + str(netw_i) + "_instance/"
    traff_pathname = str(lpMat_i) + "_traff_matrix/" + str(int(maxUtilPerNodePair[0]*100)) + "_ut/"
    #trace_" + str(trace_id) + "/"
    newDS_pathname = topo_pathname + traff_pathname
    fullDSpathnme = FLAGS.dataset_path + "state_" +str(FLAGS.state) + "/" + newDS_pathname
    #algo_name = "ILP_oracle"
    algo_name = "RWA_SPF_FF"

#    #list_dsi = []
#    # pour chaque taille de réseau (10,50,100,...)
#    for nod in nodes:
#        # pour chaque degrée (3,5,8)
#        for deg in degree:
#            # nombre maximun de liens
#            #numLinks = deg*nod
#            # pour chaque % de congestion (0,0.25,0.5)
#            for con in congestion:

    # Initialize session
    sess = tf.Session()
    K.set_session(sess)

    maxNumNodes = max(nodes)
    maxNumLinks = max(degree)*maxNumNodes
    if (FLAGS.state == 0):
        STATE_SIZE = maxNumLinks*FLAGS.numWavelengths + maxNumNodes
    else:
        STATE_SIZE = (1 + maxNumLinks*FLAGS.numWavelengths) * maxNumNodes
    print(" STATE_SIZE ", STATE_SIZE)

    # Create neural_networks
    print('instantiation of the network')
    if FLAGS.which_nn_use == 'SL_FC_LinkToPath':
        NN = SL_FC_LinkToPath(STATE_SIZE)
    elif FLAGS.which_nn_use == 'NANE_LinkToPath':
        NN = NANE_LinkToPath(STATE_SIZE, maxNumNodes, maxNumLinks)
#    elif FLAGS.which_nn_use == 'benchmarkLinkToPath':
#         make_benchmarkLinkToPath(training_data, validation_data, testing_data, variable_writer)
#
    print("NN run \n\n\n\n\n\n\n\n\n\n ")
    # Writer that allow to plot the loss and accuracy fonction in tensorboard
    merged = tf.summary.merge_all()
    variable_writer = tf.summary.FileWriter(log_dir(nodes[0], degree[0], netw_i, lpMat_i, maxUtilPerNodePair[0], traffic, algo_name), sess.graph)
    # Init variables
    sess.run(tf.global_variables_initializer())
    runSession(sess, NN, merged, variable_writer, fullDSpathnme, algo_name)

    variable_writer.flush()
    show_params()

def runSession(sess, NN, merged, variable_writer, pathname, algo_name):

    i = 0
    itN = 0
    iterSave = 0
    #Computing benchmark: Shortest Path First and First Wavelength
    trivialSol = np.zeros((1, FLAGS.k_path_generated*FLAGS.numWavelengths+1))
    trivialSol[0][1] = 1

    train_data, valid_data, test_data = loadData(pathname, algo_name)

    # trainning data
#    X_train = train_data["X"]
#    Y_train = train_data["y"]
#    topology = train_data["topology"]
#        print(topology)
#        print(len(topology))
#    valid_data = valid_data
#    test_data = test_data

    # length of the training
#    Training_length = len(X_train)

    batchSize = FLAGS.batch_size
    nbEpoch = len(train_data)
    FLAGS.epochs = nbEpoch
#    print(type(train_data))
#    print(len(train_data['0']["X"]))
#    aux_train_data = train_data['0']
#    print(aux_train_data)
#    print(aux_train_data["X"])
#    print(len(aux_train_data["X"]))
    Training_length = len(train_data['0']["X"])

 #   nbEpoch = FLAGS.epochs
    run_metadata = None

    # number of batch to use
    nbBatch = Training_length // batchSize
    print('nbBatch: ', nbBatch)

    epoch_val_costs = []
#            itN = 0
#            iterSave = 0
    for it_epoch in range(nbEpoch):

#        if (FLAGS.is_shuffling):
#            X_train, Y_train = shuffle(X_train, Y_train)

        print('Current epoch : ', (it_epoch + 1))
        X_train = train_data[str(it_epoch)]["X"]
        Y_train = train_data[str(it_epoch)]["y"]
        topology = train_data[str(it_epoch)]["topology"]
        for it_batch in range(nbBatch):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.

            itN += batchSize

            if (FLAGS.show_metadata):
                run_metadata = tf.RunMetadata()

            Batch_X = X_train[it_batch * batchSize: batchSize + it_batch * batchSize]
            Batch_Y = Y_train[it_batch * batchSize: batchSize + it_batch * batchSize]
            #batch_topology = topology[it_batch * batchSize: batchSize + it_batch * batchSize]

            #Computing benchmark: Shortest Path
            bm_accuracy = accuracyLinkToPath(Batch_Y, np.ones((batchSize, 1))*trivialSol)
            summary = tf.Summary()
            summary.value.add(tag='bm_accuracy_training', simple_value=bm_accuracy)
            variable_writer.add_summary(summary, itN)

            epoch_val_costs.append(0)
            outs, loss, predictions = NN.train(sess, NN, Batch_X, Batch_Y, topology, merged, run_metadata)
            print(predictions)
            if (it_batch + 1) % FLAGS.save_every == 0:
                # itN = it_epoch * nbBatch * batchSize + (it_batch +1) * batchSize
                NN.saveExtraData(itN, variable_writer, Batch_Y, loss, predictions)
                if (FLAGS.show_metadata):
                    NN.show_metadata(iterSave, run_metadata, variable_writer)
                    iterSave += 1

        print("Validation")
        NN.validation(sess, valid_data, NN, variable_writer, i, it_epoch, nbEpoch, nbBatch)
        #Computing benchmark: Shortest Path
        bm_accuracy = accuracyLinkToPath(valid_data["y"], np.ones((len(valid_data["y"]), 1))*trivialSol)
        summary = tf.Summary()
        summary.value.add(tag='bm_accuracy_validation', simple_value=bm_accuracy)
        variable_writer.add_summary(summary, itN)

    print("Test")
    NN.test(sess, test_data, NN, variable_writer, i)
    #Computing benchmark: Shortest Path
    bm_accuracy = accuracyLinkToPath(test_data["y"],np.ones((len(test_data["y"]), 1))*trivialSol)
    summary = tf.Summary()
    summary.value.add(tag='bm_accuracy_test', simple_value=bm_accuracy)
    variable_writer.add_summary(summary, itN)

def show_params():
    total = 0
    for v in tf.trainable_variables():
        dims = v.get_shape().as_list()
        num = int(np.prod(dims))
        total += num
        print('  %s \t\t Num: %d \t\t Shape %s ' % (v.name, num, dims))
    print('\nTotal number of params: %d' % total)


def loadData(pathname, algo_name):
    print("Loading data sets ...")
    training_data = json.load(open(pathname + algo_name + "_train.json"))
    testing_data = json.load(open(pathname + algo_name + "_test.json"))
    valid_data = json.load(open(pathname + algo_name + "_valid.json"))

    return training_data, valid_data, testing_data

main()
