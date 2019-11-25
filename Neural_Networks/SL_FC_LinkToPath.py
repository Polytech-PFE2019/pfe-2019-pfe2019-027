import pickle

import tensorflow as tf

from Data_Set.Generation.Topology import traffic_s_d_to_index
from Neural_Networks.ModelAnalyser import ModelAnalyser
from utils.proc import accuracyLinkToPath

flags = tf.app.flags
FLAGS = flags.FLAGS
from sklearn.utils import shuffle


class SL_FC_LinkToPath():

    def __init__(self, STATE_SIZE):
        # s_size: input dimension size (state size)
        # a_size: output dimension size (action size)
        self.s_size = STATE_SIZE
        self.a_size = FLAGS.k_path_generated*FLAGS.numWavelengths + 1

        # labels placeholder
        self.labels = tf.placeholder(tf.float32, [None, self.a_size], 'Y_labels')
        # input placeholder
        self.x = tf.placeholder(tf.float32, [None, self.s_size], 'X')

        # TODO: which Optimizer to use
        self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        '''
        training para in tf.layers.batch_normalization to control training/inference mode
        during training: is_train = True, normalized with statistics of the current batch
        during validation/test: is_train = False, normalized with normalized with moving statistics
        '''
        self.is_train = tf.placeholder(tf.bool, name="is_train")

        # Activation function
        self.act = self.getActivation()

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

        # prediction without loss and softmax
        self.rawY = self._build_net()

        # Loss
        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.rawY)
        tf.summary.scalar('loss', self.loss)

        #  TODO accuracy

        # prediction
        self.y = tf.nn.softmax(self.rawY)

        '''
        this is where we tell Tensorflow to update the moving average of mean and variance, at training time. 
        '''
        if FLAGS.batch_normalization:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.opt_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        else:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def getActivation(self):

        if FLAGS.activation == 'sigmoid':
            return tf.sigmoid
        elif FLAGS.activation == 'relu':
            return tf.nn.relu
        elif FLAGS.activation == 'relu6':
            return tf.nn.relu6
        else:
            return tf.nn.relu

    def _build_net(self):

        layer = self.x
        for i in range(FLAGS.nb_layers - 1 ):
            layer = self.llfc(layer, FLAGS.size_layer, True, i)
            self.variableData['layer'].append(layer)
                    
        #last fully connected layer
        layer = self.llfc(layer , self.a_size, False, FLAGS.nb_layers-1)
        self.variableData['layer'].append(layer)

        return layer

    def llfc(self, prev_layer, num_units, isAct, it):

        # initializer
        w_init = tf.random_normal_initializer(0., .1)

        with tf.name_scope("Layer_" + str(it)):

            # if we use batch norm
            if (FLAGS.batch_normalization):
                # BN
                with tf.name_scope("BN_" + str(it)):
                    LLBatchNormalisation = tf.layers.batch_normalization(prev_layer, center=True, scale=True,
                                                                         training=self.is_train)

                # weigh, bias and operation
                with tf.name_scope("OP_" + str(it)):
                    in_dim = LLBatchNormalisation.get_shape()[1]
                    wll = tf.get_variable('w_ll_' + str(it), dtype=tf.float32, shape=[in_dim, num_units],
                                          initializer=w_init)
                    bll = tf.get_variable("b_ll_" + str(it), dtype=tf.float32, shape=[num_units],
                                          initializer=tf.zeros_initializer())

                    llRes = tf.matmul(LLBatchNormalisation, wll)
                    llRes += bll

                    # activation function
                    if (isAct):
                        llRes = self.act(llRes)
                    #     self.variableData['activation'].append(self.act)
                    # else:
                    #     self.variableData['activation'].append(None)

                self.variableData['batchnorm'].append(LLBatchNormalisation)

            else:
                # weigh, bias and operation
                with tf.name_scope("OP_" + str(it)):
                    in_dim = prev_layer.get_shape()[1]
                    wll = tf.get_variable('w_ll_' + str(it), dtype=tf.float32, shape=[in_dim, num_units],
                                          initializer=w_init)
                    bll = tf.get_variable("b_ll_" + str(it), dtype=tf.float32, shape=[num_units],
                                          initializer=tf.zeros_initializer())

                    llRes = tf.matmul(prev_layer, wll)
                    llRes += bll

                    if (isAct):
                        llRes = self.act(llRes)
                    #     self.variableData['activation'].append(self.act)
                    # else:
                    #     self.variableData['activation'].append(None)

#                self.variableData['batchnorm'].append(None)
            self.variableData['weight'].append(wll)
            self.variableData['bias'].append(bll)

        return llRes


    def train(self, sess, NN, Batch_X, Batch_Y, topology, merged, run_metadata):

        # Training step
        if (FLAGS.show_metadata):
            sess.run(NN.opt_op, {NN.x: Batch_X, NN.labels: Batch_Y, NN.is_train: True},
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
            outs, loss, predictions = sess.run([merged, NN.loss, NN.y],
                                               {NN.x: Batch_X, NN.labels: Batch_Y, NN.is_train: True})
        else:
            sess.run(NN.opt_op, {NN.x: Batch_X, NN.labels: Batch_Y, NN.is_train: True},
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
            outs, loss, predictions = sess.run([merged, NN.loss, NN.y],
                                               {NN.x: Batch_X, NN.labels: Batch_Y, NN.is_train: True})

        return outs, loss, predictions

    ''' 
    if (FLAGS.show_metadata):
        sess.run(NN.opt_op, {NN.x: Batch_X, NN.labels: Batch_Y, NN.is_train: True}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
        outs, loss, predictions = sess.run([merged, NN.loss, NN.y], {NN.x: Batch_X, NN.labels: Batch_Y, NN.is_train: True})
    '''

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
        
        test_loss, test_predictions, variable = sess.run([NN.loss, NN.y, NN.variableData],
                                                             {NN.x: X_test, NN.labels: Y_test, NN.is_train: False})

        test_accuracy = accuracyLinkToPath(Y_test, test_predictions)

        # save loss in tensorboard
        summary = tf.Summary()
        summary.value.add(tag='loss_test', simple_value=test_loss)
        variable_writer.add_summary(summary, netw_inst+1)

        # save accuracy in tensorboard
        summary = tf.Summary()
        summary.value.add(tag='accuracy_test', simple_value=test_accuracy)
        variable_writer.add_summary(summary, netw_inst+1)

#        print("len(variable['layer'])" , len(variable['layer']))
#        variable['prediction'].append(test_predictions)
#        for i in range(len(variable['layer'])):
#
#            print('len ', i, ' layer ', len(variable['layer'][0][i]))

#        analyser = ModelAnalyser(variable,test_accuracy )
#        analyser.lda_labels()
#
#
#        # analyser.lda3()
#        # save loss in tensorboard
#        summary = tf.Summary()
#        summary.value.add(tag='loss_test', simple_value=test_loss)
#        variable_writer.add_summary(summary, 1)
#
#        # save accuracy in tensorboard
#        summary = tf.Summary()
#        summary.value.add(tag='accuracy_test', simple_value=test_accuracy)
#        variable_writer.add_summary(summary, 1)
#
#        # if(FLAGS.print_data) :
#        #     self.printLastLayer(test_lastLayer,test_data_readable_input,Y_test,variable_writer)

        return test_loss, test_accuracy

    def printLastLayer(self, lastLayer, test_data_readable_input, output, variable_writer):

        input = test_data_readable_input['X_test']

        with open("Data_Set/" + FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/matriceShortestPath.txt",
                  'rb') as f:
            resultMatrix = pickle.load(f)

        for i in range(0, 50):
            str_result = " Input " + str(i) + " " + str(input[i]) + "\n"

            summary = tf.Summary()
            text_tensor = tf.make_tensor_proto(str_result, dtype=tf.string)
            meta = tf.SummaryMetadata()
            meta.plugin_data.plugin_name = "text"
            summary.value.add(tag=str(i) + " Input ", metadata=meta, tensor=text_tensor)
            variable_writer.add_summary(summary, i)

            source = int(input[i][FLAGS.network_link])
            destination = int(input[i][FLAGS.network_link + 1])

            str_result = " Output :" + str(i) + " " + str(output[i]) + "\n paths : " + str(
                resultMatrix[traffic_s_d_to_index(source, destination)])

            summary = tf.Summary()
            text_tensor = tf.make_tensor_proto(str_result, dtype=tf.string)
            meta = tf.SummaryMetadata()
            meta.plugin_data.plugin_name = "text"
            summary.value.add(tag=str(i) + " Output ", metadata=meta, tensor=text_tensor)
            variable_writer.add_summary(summary, i)

            str_result = " Last Layer " + str(i) + " " + str(lastLayer[i]) + "\n\n\n "

            summary = tf.Summary()
            text_tensor = tf.make_tensor_proto(str_result, dtype=tf.string)
            meta = tf.SummaryMetadata()
            meta.plugin_data.plugin_name = "text"
            summary.value.add(tag=str(i) + " Last Layer ", metadata=meta, tensor=text_tensor)
            variable_writer.add_summary(summary, i)

    def validation(self, sess, valid_data, NN, variable_writer, netw_inst, it_epoch, nbEpoch, nbBatch):
        # Test avec le validation test

        batchSize = FLAGS.batch_size
#        X_valid = valid_data["X"][-batchSize:]
#        Y_valid = valid_data["y"][-batchSize:]
        X_valid = valid_data["X"]
        Y_valid = valid_data["y"]
        
        valid_loss, valid_predictions = sess.run([NN.loss, NN.y],
                                                     {NN.x: X_valid, NN.labels: Y_valid, NN.is_train: False})

        valid_accuracy = accuracyLinkToPath(Y_valid, valid_predictions)

        # save loss in tensorboard
        summary = tf.Summary()
        summary.value.add(tag='loss_validation', simple_value=valid_loss)
        variable_writer.add_summary(summary, (netw_inst*nbEpoch + it_epoch+1) * nbBatch * FLAGS.batch_size)

        # save accuracy in tensorboard
        summary = tf.Summary()
        summary.value.add(tag='accuracy_validation', simple_value=valid_accuracy)
        variable_writer.add_summary(summary, (netw_inst*nbEpoch + it_epoch+1) * nbBatch * FLAGS.batch_size)
