import pickle

import tensorflow as tf
from tensorflow import keras
from Data_Set.Generation.Topology import *
from Neural_Networks.ModelAnalyser import ModelAnalyser
from utils.proc import accuracyLinkToPath
from tensorflow.contrib.kfac.python.ops.utils import kronecker_product 
#
#tf.contrib.kfac.utils.kronecker_product

flags = tf.app.flags
FLAGS = flags.FLAGS
from sklearn.utils import shuffle
import numpy as np
from scipy import sparse


# TODO check if the weight are save in variableDATA
def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All zeros."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def multi_label_hot(prediction, threshold=0.5):
    prediction = tf.cast(prediction, tf.float32)
    threshold = float(threshold)
    return tf.cast(tf.greater(prediction, threshold), tf.float32)


"""
Define a keras layer
"""


class Keras_Graphsage_Layer(keras.layers.Layer):

    def __init__(self,  embedding_in_size=3,  embedding_out_size=3, input_dim=33, output_dim=33):
        super(Keras_Graphsage_Layer, self).__init__()
        self. embedding_in_size =  embedding_in_size
        self. embedding_out_size =  embedding_out_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vars = {}
        G = Topology()
        self.matrix = {}
        self.neibhour_nodes_matrix_in, neibhour_nodes_matrix_out = get_neibhour_self_nodes_matrix_rcn(G)
        # self.matrix['rcn_in'] = tf.Variable(neibhour_nodes_matrix_in, dtype=tf.float32, trainable=False)
        # self.matrix['rcn_out'] = tf.Variable(neibhour_nodes_matrix_out, dtype=tf.float32, trainable=False)

        self.neibhour_self_nodes_matrix = get_neibhour_self_nodes_matrix_gcn(G)
        # self.matrix['gcn'] = tf.Variable(neibhour_self_nodes_matrix, dtype=tf.float32, trainable=False)
        self.neibhour_nodes_matrix = get_neibhour_nodes_matrix_mean(G)
        self.neibhour_only_self = np.eye(FLAGS.network_node, dtype=np.float32)
        # print('self.neibhour_only_self',self.neibhour_only_self)
        # print('operator_0',operator_0)
        # self.matrix['mean'] = tf.Varia,ble(neibhour_nodes_matrix, dtype=tf.float32, trainable=False)

    def build(self, input_shapes):
        '''
        weight_variables
        '''
        # if(FLAGS.aggregator == 'mean'):
        self.vars['neigh_weights'] = glorot([self. embedding_in_size, self. embedding_out_size], name='neigh_weights')
        self.vars['self_weights'] = glorot([self. embedding_in_size, self. embedding_out_size], name='self_weights')
        operator_1 = tf.linalg.LinearOperatorFullMatrix(self.neibhour_nodes_matrix)  # neighbour matrix
        operator_11 = tf.linalg.LinearOperatorFullMatrix(self.neibhour_only_self)  # neighbour matrix
        operator_2 = tf.linalg.LinearOperatorFullMatrix(self.vars['neigh_weights'])  # weight variables
        operator_22 = tf.linalg.LinearOperatorFullMatrix(self.vars['self_weights'])  # weight variables

        # print('operator_1', operator_1.shape, operator_1)
        # print('operator_2', operator_2.shape,operator_2)
        from_neighs = tf.linalg.LinearOperatorKronecker([operator_1, operator_2])
        from_neighs = from_neighs.to_dense()
        from_self = tf.linalg.LinearOperatorKronecker([operator_11, operator_22])
        from_self = from_self.to_dense()
        # print('from_self', from_self)
        # print('from_neighs', from_neighs)
        self.total_weight = tf.add_n([from_self, from_neighs])
        # super(Linear, self).build(input_shape)

    def call(self, inputs, batch_train):
        if (FLAGS.batch_normalization):
            inputs = tf.layers.batch_normalization(inputs, center=True, scale=True, training=batch_train)
            return tf.nn.relu(tf.matmul(inputs, self.total_weight))
        # print('inputs',inputs)
        # print('self.total_weight',self.total_weight)
        else:
            return tf.nn.relu(tf.matmul(inputs, self.total_weight))


class NANE_LinkToPath():

    def __init__(self, inputSize, nbNodeMax, nbLinks):
        # s_size: input dimension size (state size)
        # a_size: output dimension size (action size)
        self.in_size = inputSize
        self.out_size = FLAGS.k_path_generated*FLAGS.numWavelengths + 1
        self.nbNodeMax = nbNodeMax
        self.nbLinks = nbLinks
        self.numWavelengths = FLAGS.numWavelengths
        #for state = 0 and 1, link embedding for state =2, node embedding 
        print('FLAGS.state',FLAGS.state)
        if (FLAGS.state == 0):
            self.nbEmbeddings = nbLinks
        if (FLAGS.state == 1):
            self.nbEmbeddings = nbNodeMax
            
        print('nbNodeMax', nbNodeMax)
        print('nbLinks', nbLinks)
        #self.nbEmbeddings = nbLinks if (FLAGS.state == 0 or 1) else nbNodeMax
        print('self.nbEmbeddings', self.nbEmbeddings)
        self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self.x = tf.placeholder(tf.float32, [None, self.in_size], 'Input')
        self.labels = tf.placeholder(tf.float32, [None, self.out_size], 'Output_labels')
        self.topology = tf.placeholder(tf.float32, [ self.nbEmbeddings , self.nbEmbeddings ], 'topology')
        self.eye = tf.placeholder(tf.float32, [ self.nbEmbeddings , self.nbEmbeddings ], 'Eye')
#        self.topology = tf.placeholder(tf.float32, [ self.nbNodeMax , self.nbNodeMax ], 'topology')
#        self.eye = tf.placeholder(tf.float32, [ self.nbNodeMax , self.nbNodeMax ], 'Eye')
        #self.prediction = tf.placeholder(tf.float32, [None,self.out_size ], 'Prediction')

        '''
        training para in tf.layers.batch_normalization to control training/inference mode
        during training: is_train = True, normalized with statistics of the current batch
        during validation/test: is_train = False, normalized with normalized with moving statistics
        '''
        self.is_train = tf.placeholder(tf.bool, name="is_train")

        self.loss = 0
        self.accuracy = 0
        self.tf_accuracy = 0
        self.tf_metric = 0
        self.LLBatchNormalisation = []
        
        self.variableData = {
            "weight": [],
            "bias": [],
            # "activation": [],
            "batchnorm": [],
            "layer": [],
            "labels": [],
            "prediction": []
        }

        # the first layer is the input
        self.variableData['layer'].append(self.x)
        self.variableData['labels'].append(self.labels)
        self.vars = {}
        
        #The topology is in the dataset
        self.agg_matrix = {}
        
        ######## GCN AGGREGATION ##################
        # get mean_neibhour_matrix
        print("topology ",self.topology.get_shape())
        print("eye ",self.eye.get_shape())
        aux = self.topology+self.eye
        print("self.topology+self.eye ",aux.get_shape())
        self.GCN_mean_vector = tf.reduce_mean(self.topology+self.eye,axis = 0)
        print("GCN_mean_vector ",self.GCN_mean_vector.get_shape())
        #tile the mean vector
        #multiply = tf.constant([self.nbNodeMax, 1])
        #print("self.nbNodeMax",self.nbNodeMax)
        #print("multiply ",multiply.get_shape())

        #self.GCN_mean_mask = tf.tile(self.GCN_mean_vector, [self.nbNodeMax, 1])
        self.GCN_mean_mask  = tf.ones([self.nbEmbeddings, 1]) * self.GCN_mean_vector
        print("GCN_mean_mask ",self.GCN_mean_mask.get_shape())
        #neighAggr_matrix = topology * mean_mask
        self.GCN_neighAggr_matrix = tf.multiply(self.topology+self.eye, self.GCN_mean_mask) 
        print("GCN_neighAggr_matrix ", self.GCN_neighAggr_matrix.get_shape())

        #self.matrix['mean'] = tf.Variable(self.neibhour_matrix, dtype=tf.float32, trainable=False, validate_shape=False)
        self.agg_matrix['gcn'] =  self.GCN_neighAggr_matrix
        print(" self.matrix['gcn']  ",  self.agg_matrix['gcn'] .get_shape())
        
        ######## GraphSAGE AGGREGATION ##################
        # get mean_neibhour_matrix
        self.GSAGE_mean_vector = tf.reduce_mean(self.topology,axis = 0)
        print("GSAGE_mean_vector ",self.GSAGE_mean_vector.get_shape())
        #tile the mean vector
        self.GSAGE_mean_mask = tf.ones([self.nbEmbeddings, 1]) * self.GSAGE_mean_vector
        #neighAggr_matrix = topology * mean_mask
        self.GSAGE_neighAggr_matrix = tf.multiply(self.topology, self.GSAGE_mean_mask) 
                
        print("GSAGE_neighAggr_matrix ", self.GSAGE_neighAggr_matrix.get_shape())
        #self.neibhour_only_self = tf.Variable( self.eye , dtype=tf.float32, trainable=False, validate_shape=False)
        self.self_matrix = self.eye
        print("self_matrix ", self.self_matrix.get_shape())
        #self.matrix['mean'] = tf.Variable(self.neibhour_matrix, dtype=tf.float32, trainable=False, validate_shape=False)
        
        self.agg_matrix['mean'] =  self.GSAGE_neighAggr_matrix
        print(" self.matrix['mean']  ",  self.agg_matrix['mean'] .get_shape())
      
        ######## R-GCN AGGREGATION ##################
        self.RGCN_out_mean_vector = tf.reduce_mean(tf.transpose(self.topology) ,axis = 0)
        print("RGCN_out_mean_vector ",self.RGCN_out_mean_vector.get_shape())
        #tile the mean vector
        self.RGCN_out_mean_mask = tf.ones([self.nbEmbeddings, 1]) * self.RGCN_out_mean_vector
        #neighAggr_matrix = topology * mean_mask
        self.RGCN_out_neighAggr_matrix = tf.multiply(self.topology, self.RGCN_out_mean_mask) 
        print("RGCN_out_neighAggr_matrix ", self.RGCN_out_neighAggr_matrix.get_shape())
        self.RGCN_in_neighAggr_matrix = self.GSAGE_neighAggr_matrix
        
        self.agg_matrix['rgcn_in'] =  self.RGCN_in_neighAggr_matrix
        print(" self.matrix['rgcn_in']  ",  self.agg_matrix['rgcn_in'] .get_shape())
        self.agg_matrix['rgcn_out'] =  self.RGCN_out_neighAggr_matrix
        print(" self.matrix['rgcn_out']  ",  self.agg_matrix['rgcn_out'] .get_shape())
        
         ######## AGGREGATION BY MASKING DENSE W MATRIX WITH ADJACENCY MATRIX ##################
         # WARNING IN THIS CASE THE AGGR MATRIX IS num_nodes*embedding_in_size x num_nodes*embedeing_out_size 
         # INSTEAD embedding_in_size x embedeing_out_size 

        

        # Todo: change activation function for grahsage
        if FLAGS.activation == 'sigmoid':
            self.act = tf.sigmoid
        elif FLAGS.activation == 'relu':
            self.act = tf.nn.relu
        elif FLAGS.activation == 'relu6':
            self.act = tf.nn.relu6

        self.y = self._build_net()
        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.y)
        self.yy = tf.nn.softmax(self.y)
        tf.summary.scalar('loss', self.loss)
    
        '''
        this is where we tell Tensorflow to update the moving average of mean and variance, at training time. 
        '''
        if FLAGS.batch_normalization:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.opt_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        else:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def _build_net(self):   
        # Graphsage neural networks
        if (FLAGS.state == 0): # link based
            link_list = [i for i in range(self.in_size)]
            # print('self.x',self.x)
            divid_x_graphsage = tf.nn.embedding_lookup(tf.transpose(self.x), link_list[:self.nbLinks*self.numWavelengths])
            s_d_v = tf.nn.embedding_lookup(tf.transpose(self.x), link_list[self.nbLinks*self.numWavelengths:])
            self.final_embedding = self.NANE_layers(tf.transpose(divid_x_graphsage), self.numWavelengths,
                                                                                  FLAGS.nb_nane_layers,
                                                                                  self.act)
            self.x_nane = tf.concat([self.final_embedding, tf.transpose(s_d_v)], axis=1)

        else: # node based
#            if (FLAGS.masked_dense) :
#                self.x_nane = self.x
#                for i in range(FLAGS.nb_nane_layers): 
#                    self.x_nane = self.bn_masked_fc_layer(self.x_nane, self.nbEmbeddings*FLAGS.hl_embedding_size, True, i) 
#            else :     
            self.x_nane = self.NANE_layers(self.x, self.in_size // self.nbNodeMax, FLAGS.nb_nane_layers, self.act) 


        fc_layer = self.x_nane        
        for i in range(FLAGS.nb_fc_layers-1): 
            fc_layer = self.bn_fc_layer(fc_layer, FLAGS.size_layer, True, FLAGS.nb_nane_layers + i)

        #print('fc_layer.shape  ', fc_layer.shape ) 
        y = self.bn_fc_layer(fc_layer, self.out_size,False, FLAGS.nb_fc_layers+FLAGS.nb_nane_layers-1)# final layer
        
        self.variableData['layer'].append(y)
        return y
    
    def bn_dense_layers(self, num_hidden_layer, prev_layer, num_units, activation_fn, activation_fun, nameLayer):

        """
        Create a fully connected neural network of num_hidden_layer damse ;ayersusing the tensorflow impelemntation of
        a fully conected (dense) layer 
        Before each dense layer, a batch normalisation layer is implemented
        """
        w_init = tf.random_normal_initializer(0., .1)
        layer = prev_layer
        for i in range(num_hidden_layer):
            with tf.name_scope(nameLayer + "_" + str(i + 1)):
                if (FLAGS.batch_normalization):
                    layer = tf.layers.batch_normalization(layer, center=True, scale=True, training=self.is_train)
                    layer = tf.layers.dense(layer, num_units, kernel_initializer=w_init, use_bias=True, activation=None)

                    if (activation_fn):
                        layer = activation_fun(layer)
                else:
                    layer = tf.layers.dense(layer, num_units, kernel_initializer=w_init, use_bias=True,
                                            activation=None)
                    if (activation_fn):
                        layer = activation_fun(layer)
        return layer

    def bn_fc_layer (self, prev_layer, num_units, isAct, it):

        """
        Create a single dense (fully connected) layer by multipling the matrices INPUT x WEIGHTS,
        Before the dense layer, a batch normalisation (BN) layer is implemented
        Thanks to this version we can give names to the weights that we will collevt for for tensorboard
        """
        # initializer
        w_init = tf.random_normal_initializer(0., .1)
        with tf.name_scope("Layer_" + str(it)):
            # if we use batch norm
            if (FLAGS.batch_normalization):
                # BN
                with tf.name_scope("BN_" + str(it)):
                    self.LLBatchNormalisation = tf.layers.batch_normalization(prev_layer, center=True, scale=True,
                                                                         training=self.is_train)
                # weigh, bias and operation
                with tf.name_scope("OP_" + str(it)):
                    in_dim = self.LLBatchNormalisation.get_shape()[1]
                    self.wll = tf.get_variable('w_ll_' + str(it), dtype=tf.float32, shape=[in_dim, num_units],
                                          initializer=w_init)
                    self.bll = tf.get_variable("b_ll_" + str(it), dtype=tf.float32, shape=[num_units],
                                          initializer=tf.zeros_initializer())
                    self.llRes = tf.matmul(self.LLBatchNormalisation, self.wll)
                    self.llRes += self.bll
                    # activation function
                    if (isAct):
                        self.llRes = self.act(self.llRes)
                self.variableData['batchnorm'].append(self.LLBatchNormalisation)
            else:
                # weigh, bias and operation
                with tf.name_scope("OP_" + str(it)):
                    in_dim = prev_layer.get_shape()[1]
                    self.wll = tf.get_variable('w_ll_' + str(it), dtype=tf.float32, shape=[in_dim, num_units],
                                          initializer=w_init)
                    self.bll = tf.get_variable("b_ll_" + str(it), dtype=tf.float32, shape=[num_units],
                                          initializer=tf.zeros_initializer())

                    self.llRes = tf.matmul(prev_layer, self.wll)
                    self.llRes += self.bll

                    if (isAct):
                        self.llRes = self.act(self.llRes)
                        
            self.variableData['weight'].append(self.wll)
            self.variableData['bias'].append(self.bll)

        return self.llRes
    
    def bn_masked_fc_layer(self, prev_layer, num_units, isAct, it):
        # initializer
        w_init = tf.random_normal_initializer(0., .1)
        with tf.name_scope("Layer_" + str(it)):
            # if we use batch norm
            if (FLAGS.batch_normalization):
                # BN
                with tf.name_scope("BN_" + str(it)):
                    self.LLBatchNormalisation = tf.layers.batch_normalization(prev_layer, center=True, scale=True,
                                                                         training=self.is_train)
                # weigh, bias and operation
                with tf.name_scope("OP_" + str(it)):
                    in_dim = self.LLBatchNormalisation.get_shape()[1]
                    self.wll = tf.get_variable('w_ll_' + str(it), dtype=tf.float32, shape=[in_dim, num_units],
                                          initializer=w_init)
                    mask = kronecker_product(self.topology+self.eye, tf.ones([in_dim//self.nbEmbeddings, FLAGS.hidden_embedding_size]))
                    self.wll = mask*self.wll
                    self.bll = tf.get_variable("b_ll_" + str(it), dtype=tf.float32, shape=[num_units],
                                          initializer=tf.zeros_initializer())

                    self.llRes = tf.matmul(self.LLBatchNormalisation, self.wll)
                    self.llRes += self.bll
                    # activation function
                    if (isAct):
                        self.llRes = self.act(self.llRes)

                self.variableData['batchnorm'].append(self.LLBatchNormalisation)
            else:
                # weigh, bias and operation
                with tf.name_scope("OP_" + str(it)):
                    in_dim = prev_layer.get_shape()[1]
                    self.wll = tf.get_variable('w_ll_' + str(it), dtype=tf.float32, shape=[in_dim, num_units],
                                          initializer=w_init)
                    mask = kronecker_product(self.topology+self.eye, tf.ones([in_dim//self.nbEmbeddings, FLAGS.hl_embedding_size]))
                    self.wll = mask*self.wll             
                    self.bll = tf.get_variable("b_ll_" + str(it), dtype=tf.float32, shape=[num_units],
                                          initializer=tf.zeros_initializer())
                    self.llRes = tf.matmul(prev_layer, self.wll)
                    self.llRes += self.bll

                    if (isAct):
                        self.llRes = self.act(self.llRes)
 
            self.variableData['weight'].append(self.wll)
            self.variableData['bias'].append(self.bll)

        return self.llRes

    def NANE_layers(self, embeddings_in, embedding_in_size, nb_nane_layers, activation):
        #embeddings_in = tf.reshape(embeddings_in,[-1, self.nbEmbeddings, embedding_in_size])
        for i in range(nb_nane_layers):
            if i > 0 : # excepting first nane layer
                embedding_in_size = FLAGS.hidden_embedding_size
            if i == nb_nane_layers-1 :  # last layer
                embedding_out_size = FLAGS.final_embedding_size
            else :  # excepting last nane layer
                embedding_out_size = FLAGS.hidden_embedding_size
            embeddings_out = self.NANE_layer(embeddings_in, embedding_in_size, embedding_out_size, activation, i)
            embeddings_in = embeddings_out
            self.variableData['layer'].append(embeddings_out)
        #self.output_embedding = tf.reshape(embeddings_out,[-1, self.nbEmbeddings*FLAGS.embedding_size])
        self.output_embedding = embeddings_out
        return self.output_embedding
    
    
    def NANE_layer (self, embeddings_in, embedding_in_size, embedding_out_size, act_fun, it):        
        if (FLAGS.batch_normalization):
            with tf.name_scope("BN_" + str(it)):
                embeddings_in= tf.layers.batch_normalization(embeddings_in, center=True, scale=True, training=self.is_train)
                self.variableData['batchnorm'].append(embeddings_in)

        with tf.name_scope("NANE_layer" + str(it)):  
            
            with tf.name_scope("adj_masked_aggr_" + str(it)):
                if (FLAGS.aggregator == 'adj_masked_dense'):
                    mask = kronecker_product(self.topology+self.eye, tf.ones([embedding_in_size, embedding_out_size]))
                    self.vars['weights']  = glorot([ self.nbEmbeddings*embedding_in_size, self.nbEmbeddings*embedding_out_size], name='weights')    
                    self.vars['masked_weights'] = mask*self.vars['weights']
                    total_weight = self.vars['masked_weights'] 
            
            with tf.name_scope("mean_aggr_" + str(it)):
                if (FLAGS.aggregator == 'mean'):
                    self.vars['neigh_weights'] = glorot([ embedding_in_size, embedding_out_size],
                                                        name='neigh_weights')
                    self.vars['self_weights'] = glorot([ embedding_in_size, embedding_out_size],
                                                       name='self_weights')
                    from_neigh_weights = kronecker_product(self.agg_matrix['mean'], self.vars['neigh_weights'])
                    from_self_weights = kronecker_product(self.self_matrix, self.vars['self_weights'])
                    
#                    operator_1 = tf.linalg.LinearOperatorFullMatrix(self.agg_matrix['mean'])  # neighbour matrix
#                    operator_11 = tf.linalg.LinearOperatorFullMatrix(self.self_matrix)  # neighbour matrix
#                    operator_2 = tf.linalg.LinearOperatorFullMatrix(self.vars['neigh_weights'])  # weight variables
#                    operator_22 = tf.linalg.LinearOperatorFullMatrix(self.vars['self_weights'])  # weight variables
#                    
#                    from_neigh_weights = tf.linalg.LinearOperatorKronecker([operator_1, operator_2]).to_dense()                
#                    from_self_weights = tf.linalg.LinearOperatorKronecker([operator_11, operator_22]).to_dense()

                    total_weight = tf.add_n([from_self_weights, from_neigh_weights])
                    
#                    print('from_self\n\n\n\n ', from_self.get_shape())
#                    print('from_neighs \n\n\n\n ', from_neighs.get_shape())
#                    print("total_bias ",total_bias.get_shape())
                        
            with tf.name_scope("gcn_aggr_" + str(it)):
                if (FLAGS.aggregator == 'gcn'):
                    self.vars['weights'] = glorot([ embedding_in_size, embedding_out_size],
                                                  name='weights')
                    total_weight = kronecker_product(self.agg_matrix['gcn'], self.vars['weights'])

#                    operator_1 = tf.linalg.LinearOperatorFullMatrix(self.agg_matrix['gcn'])  # neighbour matrix
#                    operator_2 = tf.linalg.LinearOperatorFullMatrix(self.vars['weights'])  # weight variables
#                    total_weight = tf.linalg.LinearOperatorKronecker([operator_1, operator_2]).to_dense()
#
            with tf.name_scope("rgcn_aggr_" + str(it)):
                if (FLAGS.aggregator == 'mean_out_in'):
                    self.vars['self_weights'] = glorot([ embedding_in_size, embedding_out_size], name='self_weights')
                    self.vars['in_neigh_weights'] = glorot([ embedding_in_size, embedding_out_size], name='in_neigh_weights')
                    self.vars['out_neigh_weights'] = glorot([ embedding_in_size, embedding_out_size],
                                                            name='out_neigh_weights')
                    
                    from_neigh_in_weights = kronecker_product(self.agg_matrix['rgcn_in'], self.vars['in_neigh_weights'])
                    from_neigh_out_weights = kronecker_product(self.agg_matrix['rgcn_out'], self.vars['out_neigh_weights'])
                    from_self_weights = kronecker_product(self.self_matrix, self.vars['self_weights'])
                    
#                    operator_1 = tf.linalg.LinearOperatorFullMatrix(self.agg_matrix['rcn_in'])  # neighbour matrix
#                    operator_11 = tf.linalg.LinearOperatorFullMatrix(self.self_matrix)  # neighbour matrix
#                    operator_111 = tf.linalg.LinearOperatorFullMatrix(self.agg_matrix['rcn_out'])  # neighbour matrix
#    
#                    operator_2 = tf.linalg.LinearOperatorFullMatrix(self.vars['in_neigh_weights'])  # weight variables
#                    operator_22 = tf.linalg.LinearOperatorFullMatrix(self.vars['self_weights'])  # weight variables
#                    operator_222 = tf.linalg.LinearOperatorFullMatrix(self.vars['out_neigh_weights'])  # weight variables
#    
#                    from_neigh_in_weights = tf.linalg.LinearOperatorKronecker([operator_1, operator_2]).to_dense()
#                    from_self_weights = tf.linalg.LinearOperatorKronecker([operator_11, operator_22]).to_dense()
#                    from_neigh_out_weights = tf.linalg.LinearOperatorKronecker([operator_111, operator_222]).to_dense()
    
                    total_weight = tf.add_n([from_neigh_in_weights, from_self_weights, from_neigh_out_weights])
                
            self.variableData['weight'].append(total_weight)
            self.vars['nane_bias'] = zeros([embedding_out_size], name='nane_bias')
            total_bias  = tf.tile(self.vars['nane_bias'], [self.nbEmbeddings])
            return act_fun(tf.matmul(embeddings_in, total_weight) + total_bias)

    def train(self, sess, NN, Batch_X, Batch_Y, topology_inc, merged, run_metadata):
    
        #for state = 0 and 1, link embedding
        if (FLAGS.state == 0):
            #topology from data is an incidence matrix (links vs nodes)
            # for link embedding, we transform it into the adjacency matrix of the transformed line node graph (links --> nodes)
            #print("topology_inc ",topology_inc)
            link_vs_link_mat = -np.matmul(topology_inc, np.transpose(topology_inc))
            topology_adj = np.clip(np.sign(link_vs_link_mat),0,1)
            eye = np.eye(self.nbLinks)
            #print("topology_adj ",topology_adj)        
        #for state =1, node embedding            
        if (FLAGS.state == 1):
            #topology from data is an incidence matrix (links vs nodes)
            # for node embedding, we transform it into the adjacency matrix of the original node graph
            #print("topology_inc ",topology_inc)
            node_vs_node_mat = -np.matmul(np.transpose(topology_inc), topology_inc)
            topology_adj = np.clip(np.sign(node_vs_node_mat),0,1)
            eye = np.eye(self.nbNodeMax)
            #print("topology_adj ",topology_adj)
        
        #print("topology ",topology)
        #print("NN.topology ",NN.topology.get_shape())
        #print("topology ",topology.get_shape())
        #batchEye = [np.eye(FLAGS.network_node) for i in range(len(Batch_Y))]
        #eye = np.eye(self.nbNodeMax)
        if (FLAGS.show_metadata):
            sess.run(NN.opt_op, {NN.x: Batch_X, NN.labels: Batch_Y, NN.is_train: True, NN.topology:topology_adj, NN.eye: eye},
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
            outs, loss, predictions = sess.run([merged, NN.loss, NN.y],
                                               {NN.x: Batch_X, NN.labels: Batch_Y, NN.is_train: True, NN.topology:topology_adj, NN.eye: eye})
#            outs, loss, predictions = sess.run([merged, NN.loss, NN.y, NN.total_weight],
#                                               {NN.x: Batch_X, NN.labels: Batch_Y, NN.is_train: True, NN.topology:topology, NN.eye: eye},
#                                               options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
        else:   
            sess.run(NN.opt_op, {NN.x: Batch_X, NN.labels: Batch_Y, NN.is_train: True, NN.topology:topology_adj, NN.eye: eye},
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
            outs, loss, predictions = sess.run([merged, NN.loss, NN.y],
                                               {NN.x: Batch_X, NN.labels: Batch_Y, NN.is_train: True, NN.topology:topology_adj, NN.eye: eye})

        return outs, loss, predictions

    def show_metadata(self, nb_iter, run_metadata, variable_writer):
        summary = tf.Summary()
        variable_writer.add_run_metadata(run_metadata, 'step%d' % nb_iter)
        variable_writer.add_summary(summary, nb_iter)

    def saveExtraData(self, itN, variable_writer, Batch_Y, loss, predictions):

        summary = tf.Summary()

        summary.value.add(tag='loss_training', simple_value=loss)
        variable_writer.add_summary(summary, itN)
        # calculate the accuracy
        Batch_accuracy = accuracyLinkToPath(Batch_Y, predictions)

        summary.value.add(tag='accuracy_training', simple_value=Batch_accuracy)
        variable_writer.add_summary(summary, itN)

    def test(self, sess, test_data, NN, variable_writer, netw_inst):
        batchSize = FLAGS.batch_size 
#        X_test = test_data["X"][-batchSize:]
#        Y_test = test_data["y"][-batchSize:]
        X_test = test_data["X"]
        Y_test = test_data["y"]
        topology_inc = test_data["topology"]
        #batchEye = [np.eye(FLAGS.network_node) for i in range(len(X_test))]
        #eye = np.eye(self.nbNodeMax)
        
        #for state = 0 and 1, link embedding
        if (FLAGS.state == 0):
            #topology from data is an incidence matrix (links vs nodes)
            # for link embedding, we transform it into the adjacency matrix of the transformed line node graph (links --> nodes)
            #print("topology_inc ",topology_inc)
            link_vs_link_mat = -np.matmul(topology_inc, np.transpose(topology_inc))
            topology_adj = np.clip(np.sign(link_vs_link_mat),0,1)
            eye = np.eye(self.nbLinks)
            #print("topology_adj ",topology_adj)        
        #for state =2, node embedding            
        if (FLAGS.state == 1):
            #topology from data is an incidence matrix (links vs nodes)
            # for node embedding, we transform it into the adjacency matrix of the original node graph
            #print("topology_inc ",topology_inc)
            node_vs_node_mat = -np.matmul(np.transpose(topology_inc), topology_inc)
            topology_adj = np.clip(np.sign(node_vs_node_mat),0,1)
            eye = np.eye(self.nbNodeMax)
            #print("topology_adj ",topology_adj)
        
        test_loss, test_predictions, test_lastLayer, ll_batchNorm, ll_weight, ll_bias, ll_beforeSofmax, variable = sess.run(
            [NN.loss, NN.y, NN.x_nane, NN.LLBatchNormalisation, NN.wll, NN.bll, NN.llRes, NN.variableData],
            {NN.x: X_test, NN.labels: Y_test, NN.is_train: False, NN.topology: topology_adj, NN.eye: eye})
        test_accuracy = accuracyLinkToPath(Y_test, test_predictions)

#        ######### LDA Analysis ###########
#        variable['prediction'].append(test_predictions)
#
#        print("nb layer : " , len(variable['layer']))
#        for i in range(len(variable['layer'])):
#            print('len ', i, ' layer ', len(variable['layer'][:][i]))
#            print('len ', i, ' layer ', len(variable['layer'][:][i][0]))
#
#        analyser = ModelAnalyser(variable,test_accuracy )
#        analyser.lda_labels()
#        analyser.lda_prediction()
#        ######### LDA Analysis ###########
        
        # save loss in tensorboard
        summary = tf.Summary()
        summary.value.add(tag='loss_test', simple_value=test_loss)
        variable_writer.add_summary(summary, netw_inst+1)

        # save accuracy in tensorboard
        summary = tf.Summary()
        summary.value.add(tag='accuracy_test', simple_value=test_accuracy)
        variable_writer.add_summary(summary, netw_inst+1)

        # if (FLAGS.print_data):
        #     self.printLastLayer(test_lastLayer, test_data_readable_input, Y_test, variable_writer, test_predictions,
        #                         ll_batchNorm, ll_weight, ll_bias, ll_beforeSofmax)

        return test_loss, test_accuracy

    def printResult(self, input, g, variable_writer, i, meanImpact, meanSquareImpact, relativeImpact,
                    relativeSquareImpact, resultImpact):
        str_result = ""

        # todo highlight the paths in the k shortes paths
        for j in range(len(g.edges())):
            str_result += " edges : " + str(g.edges()[j]) + " capacity  : " + str(input[j])
            str_result += " \n\n\n"
        for j in range(self.nbNodeMax):
            str_result += "node : " + str(j) + " resultImpact : " + str(resultImpact[j]) + " relativeImpact : " + str(
                relativeImpact[j]) + " relativeSquareImpact " + str(relativeSquareImpact[j])
            str_result += "\n\n\n"

        str_result += " meanImpact  " + str(meanImpact) + " meanSquareImpact  " + str(meanSquareImpact)
        str_result += "Source : " + str(input[FLAGS.network_link]) + "\n\n\n"
        str_result += "Destination : " + str(input[nbLinks + 1]) + "\n\n\n"
        str_result += "flow : " + str(input[FLAGS.network_link + 2]) + "\n\n\n"
        summary = tf.Summary()
        text_tensor = tf.make_tensor_proto(str_result, dtype=tf.string)
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary.value.add(tag=str(i) + " Input", metadata=meta, tensor=text_tensor)
        variable_writer.add_summary(summary, i)

    def printLastLayer(self, lastLayerGC, test_data_readable_input, output, variable_writer, test_predictions,
                       ll_batchNorm, ll_weight, ll_bias, ll_beforeSofmax):
        input = test_data_readable_input['X_test']

        g = Topology()

        with open("Data_Set/" + FLAGS.dataset_name + "/" + str(self.nbNodeMax) + "-node/matriceShortestPath.txt",
                  'rb') as f:
            resultMatrix = pickle.load(f)

        for i in range(0, 50):

            # plot the input information
            # str_result = ""
            # for j in range(len(input[i])):
            #      str_result +=  str(input[i][j]) + " <=> "  + str(g.edges()[j])  + "\n"
            # str_result +=  "Source : " + str(input[i][nbLinks])   + "\n"
            # str_result +=  "Destination : " + str(input[i][FLAGS.network_link + 1] )  + "\n"
            # str_result +=  "flow : " + str(input[i][nbLinks + 2])   + "\n"
            # summary = tf.Summary()
            # text_tensor = tf.make_tensor_proto(str_result, dtype=tf.string)
            # meta = tf.SummaryMetadata()
            # meta.plugin_data.plugin_name = "text"
            # summary.value.add(tag=str(i) + " Input", metadata=meta, tensor=text_tensor)
            # variable_writer.add_summary(summary , i )
            #

            # plot the output information
            source = int(input[i][FLAGS.network_link])
            destination = int(input[i][nbLinks + 1])

            str_result = " Output :" + str(i) + " " + str(output[i]) + "\n paths : " + str(
                resultMatrix[traffic_s_d_to_index(source, destination)])
            str_result += " test_predictions : " + str(test_predictions[i])

            summary = tf.Summary()
            text_tensor = tf.make_tensor_proto(str_result, dtype=tf.string)
            meta = tf.SummaryMetadata()
            meta.plugin_data.plugin_name = "text"
            summary.value.add(tag=str(i) + " Output ", metadata=meta, tensor=text_tensor)
            variable_writer.add_summary(summary, i)

            # plot the layers information
            str_result = " Last Layer " + str(i) + " " + str(lastLayerGC[i]) + " length \n " + str(
                len(lastLayerGC[i])) + "\n\n\n "

            lastLayerOP = np.copy(lastLayerGC[i])
            lastLayerOP = np.asmatrix(np.reshape(lastLayerOP, (self.nbNodeMax, FLAGS.embedding_size)))
            lastLayerMax = []
            lastLayerMean = []

            for j in range(0, len(lastLayerOP)):
                lastLayerMean.append(np.mean(lastLayerOP[j]))
                lastLayerMax.append(np.amax(lastLayerOP[j]))

            str_result += " lastLayerMax " + str(i) + " " + str(lastLayerMax) + " length \n " + str(
                len(lastLayerMax)) + "\n\n\n "
            str_result += " lastLayerMean " + str(i) + " " + str(lastLayerMean) + " length \n " + str(
                len(lastLayerMean)) + "\n\n\n "
            str_result += " ll_batchNorm " + str(i) + " " + str(ll_batchNorm[i]) + " length \n " + str(
                len(ll_batchNorm[i])) + "\n\n\n "
            str_result += " ll_weight " + str(i) + " " + str(ll_weight) + " length \n " + str(
                len(ll_weight)) + "\n\n\n "
            str_result += " ll_bias " + str(i) + " " + str(ll_bias) + " length \n " + str(len(ll_bias)) + "\n\n\n "
            str_result += " ll_beforeSofmax " + str(i) + " " + str(ll_beforeSofmax[i]) + " length \n " + str(
                len(ll_beforeSofmax[i])) + "\n\n\n "

            summary = tf.Summary()
            text_tensor = tf.make_tensor_proto(str_result, dtype=tf.string)
            meta = tf.SummaryMetadata()
            meta.plugin_data.plugin_name = "text"
            summary.value.add(tag=str(i) + " Last Layer ", metadata=meta, tensor=text_tensor)
            variable_writer.add_summary(summary, i)

            #  plot the scatter plot

            nodesImpacts = np.zeros((self.nbNodeMax, FLAGS.k_path_generated + 1))

            # number of nodes
            for j in range(self.nbNodeMax):

                # each final state
                for k in range(FLAGS.k_path_generated + 1):

                    # size of the embedding
                    for l in range(FLAGS.embedding_size):
                        if ll_weight[j * FLAGS.embedding_size + l][k] < 0 or lastLayerGC[i][
                            j * FLAGS.embedding_size + l] < 0:
                            nodesImpacts[j][k] += 0
                        else:
                            nodesImpacts[j][k] += ll_weight[j * FLAGS.embedding_size + l][k] * lastLayerGC[i][
                                j * FLAGS.embedding_size + l]

                    # nodesImpacts[j][k] += ll_bias[k]

            meanImpact = np.zeros(FLAGS.k_path_generated + 1)
            for j in range(FLAGS.k_path_generated + 1):
                for k in range(self.nbNodeMax):
                    meanImpact[j] += nodesImpacts[k][j]

            relativeImpact = np.zeros((self.nbNodeMax, FLAGS.k_path_generated + 1))
            for j in range(self.nbNodeMax):
                for k in range(FLAGS.k_path_generated + 1):
                    b = ll_bias[k]

                    if (b < 0 and meanImpact[k] < 0):
                        relativeImpact[j][k] = '%.2f' % 0
                    elif (b < 0):
                        relativeImpact[j][k] = '%.2f' % ((nodesImpacts[j][k]) / (meanImpact[k]))
                    elif (meanImpact[k] < 0):
                        relativeImpact[j][k] = '%.2f' % ((nodesImpacts[j][k]) / (ll_bias[k]))
                    else:
                        relativeImpact[j][k] = '%.2f' % ((nodesImpacts[j][k]) / (meanImpact[k] + ll_bias[k]))

            resultImpact = np.zeros((self.nbNodeMax, FLAGS.k_path_generated + 1))
            for j in range(self.nbNodeMax):
                for k in range(FLAGS.k_path_generated + 1):
                    resultImpact[j][k] = '%.2f' % (relativeImpact[j][k] * ll_beforeSofmax[i][k])
            meanImpact /= self.nbNodeMax
            meanImpact = ['%.2f' % elem for elem in meanImpact]
            # globalImpact = np.zeros(FLAGS.k_path_generated + 1)
            # for i in range(len(globalImpact)):
            #     for j in range(self.nbNodeMax):
            #         meanImpact[i] += nodesImpacts[j][i]
            # meanImpact = meanImpact / self.nbNodeMax

            meanSquareImpact = np.zeros(FLAGS.k_path_generated + 1, dtype=np.float64)
            for j in range(FLAGS.k_path_generated + 1):
                for k in range(self.nbNodeMax):
                    meanSquareImpact[j] += nodesImpacts[k][j] * nodesImpacts[k][j]

            relativeSquareImpact = np.zeros((self.nbNodeMax, FLAGS.k_path_generated + 1), dtype=np.float64)
            for j in range(self.nbNodeMax):
                for k in range(FLAGS.k_path_generated + 1):
                    relativeSquareImpact[j][k] = '%.2f' % (
                                (nodesImpacts[j][k] * nodesImpacts[j][k]) / meanSquareImpact[k])
            meanSquareImpact /= self.nbNodeMax
            meanSquareImpact = ['%.2f' % elem for elem in meanSquareImpact]

            str_result = "nodesImpacts " + str(nodesImpacts) + "\n\n\n"
            str_result += " meanImpact " + str(meanImpact) + "\n\n\n"
            str_result += " relativeImpact " + str(relativeImpact) + "\n\n\n"
            str_result += " meanSquareImpact " + str(meanSquareImpact) + "\n\n\n"
            str_result += " relativeSquareImpact " + str(relativeSquareImpact) + "\n\n\n"
            str_result += " resultImpact " + str(resultImpact) + "\n\n\n"
            str_result += " ll_beforeSofmax " + str(ll_beforeSofmax) + "\n\n\n"

            summary = tf.Summary()
            text_tensor = tf.make_tensor_proto(str_result, dtype=tf.string)
            meta = tf.SummaryMetadata()
            meta.plugin_data.plugin_name = "text"
            summary.value.add(tag=str(i) + "Impacts", metadata=meta, tensor=text_tensor)
            variable_writer.add_summary(summary, i)

            self.printResult(input[i], g, variable_writer, i, meanImpact, meanSquareImpact, relativeImpact,
                             relativeSquareImpact, resultImpact)

        error = 0
        for i in range(50, len(lastLayerGC)):

            if (output[i][2] == 1 or output[i][3] == 1):
                # plot the output information
                source = int(input[i][FLAGS.network_link])
                destination = int(input[i][nbLinks + 1])

                str_result = " Output :" + str(i) + " " + str(output[i]) + "\n paths : " + str(
                    resultMatrix[traffic_s_d_to_index(source, destination)])
                str_result += " test_predictions : " + str(test_predictions[i])
                if (np.argmax(output[i]) != np.argmax(test_predictions[i])):
                    error += 1
                summary = tf.Summary()
                text_tensor = tf.make_tensor_proto(str_result, dtype=tf.string)
                meta = tf.SummaryMetadata()
                meta.plugin_data.plugin_name = "text"
                summary.value.add(tag=str(i) + " Output ", metadata=meta, tensor=text_tensor)
                variable_writer.add_summary(summary, i)

                # plot the layers information
                str_result = " Last Layer " + str(i) + " " + str(lastLayerGC[i]) + " length \n " + str(
                    len(lastLayerGC[i])) + "\n\n\n "

                lastLayerOP = np.copy(lastLayerGC[i])
                lastLayerOP = np.asmatrix(np.reshape(lastLayerOP, (self.nbNodeMax, FLAGS.embedding_size)))
                lastLayerMax = []
                lastLayerMean = []

                for j in range(0, len(lastLayerOP)):
                    lastLayerMean.append(np.mean(lastLayerOP[j]))
                    lastLayerMax.append(np.amax(lastLayerOP[j]))

                str_result += " lastLayerMax " + str(i) + " " + str(lastLayerMax) + " length \n " + str(
                    len(lastLayerMax)) + "\n\n\n "
                str_result += " lastLayerMean " + str(i) + " " + str(lastLayerMean) + " length \n " + str(
                    len(lastLayerMean)) + "\n\n\n "
                str_result += " ll_batchNorm " + str(i) + " " + str(ll_batchNorm[i]) + " length \n " + str(
                    len(ll_batchNorm[i])) + "\n\n\n "
                str_result += " ll_weight " + str(i) + " " + str(ll_weight) + " length \n " + str(
                    len(ll_weight)) + "\n\n\n "
                str_result += " ll_bias " + str(i) + " " + str(ll_bias) + " length \n " + str(len(ll_bias)) + "\n\n\n "
                str_result += " ll_beforeSofmax " + str(i) + " " + str(ll_beforeSofmax[i]) + " length \n " + str(
                    len(ll_beforeSofmax[i])) + "\n\n\n "

                summary = tf.Summary()
                text_tensor = tf.make_tensor_proto(str_result, dtype=tf.string)
                meta = tf.SummaryMetadata()
                meta.plugin_data.plugin_name = "text"
                summary.value.add(tag=str(i) + " Last Layer ", metadata=meta, tensor=text_tensor)
                variable_writer.add_summary(summary, i)

                #  plot the scatter plot

                nodesImpacts = np.zeros((self.nbNodeMax, FLAGS.k_path_generated + 1))

                # number of nodes
                for j in range(self.nbNodeMax):

                    # each final state
                    for k in range(FLAGS.k_path_generated + 1):

                        # size of the embedding
                        for l in range(FLAGS.embedding_size):
                            nodesImpacts[j][k] += ll_weight[j * FLAGS.embedding_size + l][k] * lastLayerGC[i][
                                j * FLAGS.embedding_size + l]

                        nodesImpacts[j][k] += ll_bias[k]

                meanImpact = np.zeros(FLAGS.k_path_generated + 1)
                for j in range(FLAGS.k_path_generated + 1):
                    for k in range(self.nbNodeMax):
                        meanImpact[j] += nodesImpacts[k][j]

                relativeImpact = np.zeros((self.nbNodeMax, FLAGS.k_path_generated + 1))
                for j in range(self.nbNodeMax):
                    for k in range(FLAGS.k_path_generated + 1):
                        relativeImpact[j][k] = '%.2f' % (nodesImpacts[j][k] / meanImpact[k])

                resultImpact = np.zeros((self.nbNodeMax, FLAGS.k_path_generated + 1))
                for j in range(self.nbNodeMax):
                    for k in range(FLAGS.k_path_generated + 1):
                        resultImpact[j][k] = '%.2f' % (relativeImpact[j][k] * ll_beforeSofmax[i][k])
                meanImpact /= self.nbNodeMax
                meanImpact = ['%.2f' % elem for elem in meanImpact]
                # globalImpact = np.zeros(FLAGS.k_path_generated + 1)
                # for i in range(len(globalImpact)):
                #     for j in range(self.nbNodeMax):
                #         meanImpact[i] += nodesImpacts[j][i]
                # meanImpact = meanImpact / self.nbNodeMax

                meanSquareImpact = np.zeros(FLAGS.k_path_generated + 1, dtype=np.float64)
                for j in range(FLAGS.k_path_generated + 1):
                    for k in range(self.nbNodeMax):
                        meanSquareImpact[j] += nodesImpacts[k][j] * nodesImpacts[k][j]

                relativeSquareImpact = np.zeros((self.nbNodeMax, FLAGS.k_path_generated + 1), dtype=np.float64)
                for j in range(self.nbNodeMax):
                    for k in range(FLAGS.k_path_generated + 1):
                        relativeSquareImpact[j][k] = '%.2f' % (
                                    (nodesImpacts[j][k] * nodesImpacts[j][k]) / meanSquareImpact[k])
                meanSquareImpact /= self.nbNodeMax
                meanSquareImpact = ['%.2f' % elem for elem in meanSquareImpact]

                str_result = "nodesImpacts " + str(nodesImpacts) + "\n\n\n"
                str_result += " meanImpact " + str(meanImpact) + "\n\n\n"
                str_result += " relativeImpact " + str(relativeImpact) + "\n\n\n"
                str_result += " meanSquareImpact " + str(meanSquareImpact) + "\n\n\n"
                str_result += " relativeSquareImpact " + str(relativeSquareImpact) + "\n\n\n"
                str_result += " resultImpact " + str(resultImpact) + "\n\n\n"
                str_result += " ll_beforeSofmax " + str(ll_beforeSofmax) + "\n\n\n"

                # positionOfNodes = np.zeros((self.nbNodeMax * 2 ,FLAGS.k_path_generated))
                # for i in range(self.nbNodeMax) :
                #     print(" ll_weight[i][0] * ll_batchNorm[i] + ll_bias[0] + ll_weight[i][0] * ll_batchNorm[i+1] + ll_bias[0] + ll_weight[i+2][0] * ll_batchNorm[i] + ll_bias[0] ",  ll_weight[i][0] * ll_batchNorm[i] + ll_bias[0] + ll_weight[i][0] * ll_batchNorm[i+1] + ll_bias[0] + ll_weight[i+2][0] * ll_batchNorm[i] + ll_bias[0] )
                #     positionOfNodes[i][0] = ll_weight[i][0] * ll_batchNorm[i] + ll_bias[0] + ll_weight[i][0] * ll_batchNorm[i+1] + ll_bias[0] + ll_weight[i+2][0] * ll_batchNorm[i] + ll_bias[0]
                #     positionOfNodes[i + self.nbNodeMax][1] = ll_weight[i][1] * ll_batchNorm[i] + ll_bias[1] + ll_weight[i][1] * ll_batchNorm[i+1] + ll_bias[1] + ll_weight[i+2][1] * ll_batchNorm[i] + ll_bias[1]
                #     positionOfNodes[i][2] = ll_weight[i][2] * ll_batchNorm[i] + ll_bias[2] + ll_weight[i][2] * ll_batchNorm[i+1] + ll_bias[2] + ll_weight[i+2][2] * ll_batchNorm[i] + ll_bias[2]
                #     positionOfNodes[i + self.nbNodeMax][3] = ll_weight[i][3] * ll_batchNorm[i] + ll_bias[3] + ll_weight[i][3] * ll_batchNorm[i+1] + ll_bias[3] + ll_weight[i+2][3] * ll_batchNorm[i] + ll_bias[3]
                #
                #     positionOfNodes[i + self.nbNodeMax][0] = 1
                #     positionOfNodes[i ][1] = 1
                #     positionOfNodes[i + self.nbNodeMax][2] = 1
                #     positionOfNodes[i ][3] = 1
                #
                #

                # n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                #
                # str_result += "x = " + str(positionOfNodes[0]) + "\n"
                # str_result += "y = " + str(positionOfNodes[1]) + "\n"
                # str_result += "n = " + str(n) + "\n"
                # str_result += "fig, ax = plt.subplots() \n "
                # str_result += "ax.scatter(x, y) \n "
                # str_result += "for i, txt in enumerate(n): \n "
                # str_result += " ax.annotate(txt, (x[i], y[i])) \n "
                # str_result += "plt.xlabel('P1') \n "
                # str_result += "plt.xlabel('P2') \n "
                # str_result += "plt.savefig('ScatterPlot_01.png') \n "
                # str_result += "plt.show() \n "
                # str_result += "\n\n\n\n\n\n\n"
                #
                #
                # str_result += "x = " + str(positionOfNodes[2]) + "\n"
                # str_result += "y = " + str(positionOfNodes[3]) + "\n"
                # str_result += "n = " + str(n) + "\n"
                # str_result += "fig, ax = plt.subplots() \n "
                # str_result += "ax.scatter(x, y) \n "
                # str_result += "for i, txt in enumerate(n): \n "
                # str_result += " ax.annotate(txt, (x[i], y[i])) \n "
                # str_result += "plt.xlabel('P1') \n "
                # str_result += "plt.xlabel('P2') \n "
                # str_result += "plt.savefig('ScatterPlot_01.png') \n "
                # str_result += "plt.show() \n "
                #

                summary = tf.Summary()
                text_tensor = tf.make_tensor_proto(str_result, dtype=tf.string)
                meta = tf.SummaryMetadata()
                meta.plugin_data.plugin_name = "text"
                summary.value.add(tag=str(i) + "Impacts", metadata=meta, tensor=text_tensor)
                variable_writer.add_summary(summary, i)

                self.printResult(input[i], g, variable_writer, i, meanImpact, meanSquareImpact, relativeImpact,
                                 relativeSquareImpact, resultImpact)
        print("error of prediction in PCC 2 AND 3 ", error)

    def validation(self, sess, valid_data, NN, variable_writer, netw_inst, it_epoch, nbEpoch, nbBatch):
        # Test avec le validation test

        batchSize = FLAGS.batch_size
#        X_valid = valid_data["X"][-batchSize:]
#        Y_valid = valid_data["y"][-batchSize:]
        X_valid = valid_data["X"]
        Y_valid = valid_data["y"]
        
        topology_inc = valid_data["topology"]
        #batchEye = [np.eye(self.nbNodeMax) for i in range(len(X_valid))]
        #eye = np.eye(self.nbNodeMax)
   
        #for state = 0 and 1, link embedding
        if (FLAGS.state == 0):
            #topology from data is an incidence matrix (links vs nodes)
            # for link embedding, we transform it into the adjacency matrix of the transformed line node graph (links --> nodes)
            #print("topology_inc ",topology_inc)
            link_vs_link_mat = -np.matmul(topology_inc, np.transpose(topology_inc))
            topology_adj = np.clip(np.sign(link_vs_link_mat),0,1)
            eye = np.eye(self.nbLinks)
            #print("topology_adj ",topology_adj)        
        #for state =2, node embedding            
        if (FLAGS.state == 1):
            #topology from data is an incidence matrix (links vs nodes)
            # for node embedding, we transform it into the adjacency matrix of the original node graph
            #print("topology_inc ",topology_inc)
            node_vs_node_mat = -np.matmul(np.transpose(topology_inc), topology_inc)
            topology_adj = np.clip(np.sign(node_vs_node_mat),0,1)
            eye = np.eye(self.nbNodeMax)
            #print("topology_adj ",topology_adj)

        valid_loss, valid_predictions = sess.run([NN.loss, NN.y],
                                                 {NN.x: X_valid, NN.labels: Y_valid, NN.is_train: False, NN.topology: topology_adj, NN.eye: eye})

        valid_accuracy = accuracyLinkToPath(Y_valid, valid_predictions)

        # save loss in tensorboard
        summary = tf.Summary()
        summary.value.add(tag='loss_validation', simple_value=valid_loss)
        variable_writer.add_summary(summary, ((netw_inst)*nbEpoch + it_epoch+1) * nbBatch * FLAGS.batch_size)

        # save accuracy in tensorboard
        summary = tf.Summary()
        summary.value.add(tag='accuracy_validation', simple_value=valid_accuracy)
        variable_writer.add_summary(summary, (netw_inst*nbEpoch + it_epoch+1) * nbBatch * FLAGS.batch_size)