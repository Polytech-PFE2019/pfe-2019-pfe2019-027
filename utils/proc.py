import tensorflow as tf
import numpy as np
from Data_Set.Generation.Topology import Topology
from Data_Set.Generation.YenKShortestPaths import YenKShortestPaths
import os

flags = tf.app.flags
FLAGS = flags.FLAGS
from sklearn.utils import shuffle


def log_dir(nodes, degree, netw_i, mat_id, maxNodePairUtil, traffic, algo_name):
    test_log_dir =  ""
    if(FLAGS.which_nn_use == 'NANE_LinkToPath'):
            test_log_dir += "{which_nn_use:s}_{batch_normmal:b}_{activation:s}_{state:d}_{nb_layers:d}_{batch_size:d}_{agr:s}_{final_embed_size:d}_{embed_nb_layers:d}_{embed_hl_size:d}".format(
                which_nn_use=FLAGS.which_nn_use,
                batch_normmal=FLAGS.batch_normalization,
                activation=FLAGS.activation,
                state=FLAGS.state,
                nb_layers=FLAGS.nb_fc_layers,
                #size_layer=FLAGS.size_layer,
                lr=FLAGS.learning_rate,
           #     epochs=FLAGS.epochs,
                batch_size=FLAGS.batch_size,
                #max_total_steps=FLAGS.max_total_steps,
                agr=FLAGS.aggregator,
                final_embed_size=FLAGS.final_embedding_size,
                embed_nb_layers=FLAGS.nb_nane_layers,
                embed_hl_size=FLAGS.hidden_embedding_size,
                #,which_GS_use = FLAGS.which_GS_use  {which_GS_use:s}_
            )
    elif(FLAGS.which_nn_use == 'SL_FC_LinkToPath'):
            test_log_dir += "{which_nn_use:s}_{batch_normmal:b}_{activation:s}_{state:d}_{nb_layers:d}_{batch_size:d}_{size_layer:d}".format(
                which_nn_use=FLAGS.which_nn_use,
                batch_normmal=FLAGS.batch_normalization,
                activation=FLAGS.activation,
                state=FLAGS.state,
                nb_layers=FLAGS.nb_layers,
                size_layer=FLAGS.size_layer,
                lr=FLAGS.learning_rate,
           #     epochs=FLAGS.epochs,
                batch_size=FLAGS.batch_size,
           #     max_total_steps=FLAGS.max_total_steps
            )
            
    else:
            test_log_dir += "{which_nn_use:s}_{batch_normmal:b}_{activation:s}_{state:d}_{nb_layers:d}_{size_layer:d}_{lr:0.4f}_{epochs:d}_{batch_size:d}".format(
                which_nn_use=FLAGS.which_nn_use,
                batch_normmal=FLAGS.batch_normalization,
                activation=FLAGS.activation,
                state=FLAGS.state,
                nb_layers=FLAGS.nb_layers,
                size_layer=FLAGS.size_layer,
                lr=FLAGS.learning_rate,
                epochs=FLAGS.epochs,
                batch_size=FLAGS.batch_size,
                time_steps = FLAGS.time_steps
           #     max_total_steps=FLAGS.max_total_steps
            )

    FLAGS.modele_short_name = test_log_dir
    
    topo_pathname = str(nodes) + "node/" + str(int(degree)) + "degree/" + str(netw_i) + "_instance/"  
    traff_pathname = str(mat_id) + "_traff_matrix/" + str(int(maxNodePairUtil*100)) + "_ut/"
    #"trace_" + str(dynTrace_id) + "/"
    #newDS_pathname = topo_pathname + traff_pathname
    #fullDSpathnme = FLAGS.dataset_path + "state_" +str(FLAGS.state) + "/" + newDS_pathname
    #algo_name = "RWA_SPF_FF"
    
    log_dir = FLAGS.base_log_dir + topo_pathname + traff_pathname + algo_name  + '/' + test_log_dir + '/'
    FLAGS.modele_name = log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def accuracy(x, labels, y_prediction):
    num_correct_path = 0  # when is used to check if labels gives the same routing path as y_prediction
    # print('x',len(x), 'labels',len(labels),'y_prediction',len(y_prediction))
    for i in range(len(x)):
        if (FLAGS.state == 0):
            source = x[i][FLAGS.network_link]
            destination = x[i][FLAGS.network_link + 1]

        elif (FLAGS.state == 1):
            source_list = x[i][
                          2 * FLAGS.network_link * FLAGS.network_node: 2 * FLAGS.network_link * FLAGS.network_node + FLAGS.network_node]
            source = source_list.index(1.0)
            destination_list = x[i][
                               2 * FLAGS.network_link * FLAGS.network_node + FLAGS.network_node: 2 * FLAGS.network_link * FLAGS.network_node + 2 * FLAGS.network_node]
            destination = destination_list.index(1.0)
            # print('source_list', source_list, source, )
        else:
            source_destinatin_list = []
            for j in range(FLAGS.network_node):
                source_destinatin_list.append(x[i][(FLAGS.network_link + 1) * (j + 1) - 1])
            source = source_destinatin_list.index(min(source_destinatin_list))
            destination = source_destinatin_list.index(max(source_destinatin_list))

        '''
        print('i',i)
        print('x', x[i])
        print('source', source,'destination',destination)
        print('labels',labels[i])
        print('y_prediction',y_prediction[i])
        '''
        G1 = Topology()
        # print('labels : ', labels[i])
        # print('y_prediction', y_prediction[i])
        for link in range(FLAGS.network_link):
            G1[G1.edges()[link][0]][G1.edges()[link][1]]['weight'] = labels[i][link]
            # print('weight1',G1[G1.edges()[link][0]][G1.edges()[link][1]]['weight'], labels[i][link])
        G2 = Topology()
        for link in range(FLAGS.network_link):
            G2[G2.edges()[link][0]][G2.edges()[link][1]]['weight'] = y_prediction[i][link]
            # print('weight2',G2[G2.edges()[link][0]][G2.edges()[link][1]]['weight'], y_prediction[i][link])

        Paths_prediction = YenKShortestPaths(G2)
        path_prediction = (Paths_prediction.findFirstShortestPath(source, destination)).nodeList

        Paths_label = YenKShortestPaths(G1)
        path_label = (Paths_label.findFirstShortestPath(source, destination)).nodeList

        # print('path_label',path_label)
        # print('path_prediction',path_prediction)
        if path_label == path_prediction:
            num_correct_path += 1

        # print('i:',i, 'source',source,'destination',destination)

    # print('num_correct_weight',num_correct_path)
    return num_correct_path

def strauted_link_weight_error_percentage(labels, y_new_taget, y_prediction):
    # print('y_new_taget',y_new_taget)
    total_half_strauted_error_percentage = 0
    total_strauted_error_percentage = 0
    total_link_error_percentage = 0
    num_data = len(labels)
    for i in range(num_data):
        low_error = 0
        high_error = 0
        middle_error = 0
        low_weight = 0
        high_weight = 0
        middle_weight = 0
        for link in range(FLAGS.network_link):
            correct_link_weight = y_new_taget[i][link]
            if (correct_link_weight == 1):
                low_weight += 1
                if (1.1 * FLAGS.low_link_weight < y_prediction[i][link] or y_prediction[i][
                    link] < FLAGS.low_link_weight * 0.9):
                    low_error += 1
            elif (correct_link_weight == FLAGS.high_link_weight):
                high_weight += 1
                if (1.1 * correct_link_weight < y_prediction[i][link] or y_prediction[i][
                    link] < correct_link_weight * 0.9):
                    high_error += 1
            else:
                middle_weight += 1
                if (1.1 * correct_link_weight < y_prediction[i][link] or y_prediction[i][
                    link] < correct_link_weight * 0.9):
                    middle_error += 1

        if (low_weight > 0):
            total_link_error_percentage += 1.0 * low_error / low_weight
        if (high_weight > 0):
            total_strauted_error_percentage += 1.0 * high_error / high_weight
        if (middle_weight > 0):
            total_half_strauted_error_percentage += 1.0 * middle_error / middle_weight

    total_low_error_percentage = total_link_error_percentage / num_data
    total_strauted_error_percentage = total_strauted_error_percentage / num_data
    total_half_strauted_error_percentage = total_half_strauted_error_percentage / num_data

    # print('total_low_error_percentage', total_low_error_percentage, 'total_strauted_error_percentage',
    #      total_strauted_error_percentage,'total_half_strauted_error_percentage',total_half_strauted_error_percentage)
    return total_low_error_percentage, total_strauted_error_percentage, total_half_strauted_error_percentage

def link_weight_error_percentage(labels, y_prediction):
    total_low_error_percentage = 0
    total_high_error_percentage = 0
    num_data = len(labels)
    for i in range(num_data):
        low_error = 0
        high_error = 0
        low_weight = 0
        high_weight = 0
        for link in range(FLAGS.network_link):
            correct_link_weight = labels[i][link]
            if (correct_link_weight == FLAGS.low_link_weight):
                low_weight += 1
                if (1.1 * correct_link_weight < y_prediction[i][link] or y_prediction[i][
                    link] < correct_link_weight * 0.9):
                    low_error += 1
            else:
                high_weight += 1
                if (1.1 * correct_link_weight < y_prediction[i][link] or y_prediction[i][
                    link] < correct_link_weight * 0.9):
                    high_error += 1

        if (low_weight > 0):
            total_low_error_percentage += 1.0 * low_error / low_weight
        if (high_weight > 0):
            total_high_error_percentage += 1.0 * high_error / high_weight

    total_low_error_percentage = total_low_error_percentage / num_data
    total_high_error_percentage = total_high_error_percentage / num_data

    # print('total_low_error_percentage',total_low_error_percentage,'total_high_error_percentage',total_high_error_percentage)
    return total_low_error_percentage, total_high_error_percentage

def show_metadata(nb_iter, run_metadata, variable_writer):
    summary = tf.Summary()
    variable_writer.add_run_metadata(run_metadata, 'step%d' % nb_iter)
    variable_writer.add_summary(summary, nb_iter)

def saveExtraData(itN, variable_writer, Batch_X, Batch_Y,Batch_Y_target, loss, predictions):


    # if(FLAGS.print_data):
    #     1+1
        # # save batch accuracy
        # if (it_batch + 1) % 50  == 0:
        #
        #     # str_result = "Loss"
        #     # for i in  range(0,len(Batch_Y)//8):
        #     #     str_result += "\n Debut exemple " + str(i) + "\n"
        #     #     for j in range(0,len(Batch_Y[i])):
        #     #         str_result += " " + str(Batch_Y[i][j]) + " <=> " + str(predictions[i][j])
        #     #     str_result += "\n FIN exemple " + str(i) + "\n"
        #     # text_tensor = tf.make_tensor_proto(str_result, dtype=tf.string)
        #     # meta = tf.SummaryMetadata()
        #     # meta.plugin_data.plugin_name = "text"
        #     # summary.value.add(tag="whatever " + str(it_epoch) + "_" + str(it_batch), metadata=meta, tensor=text_tensor)
        #
        #     #print(str_result)
        #
        # value = "\n input : " + ''.join(str(Batch_X[1]))  + "\n label :  "+ ''.join(str(Batch_Y[1])) + "\n output "+ ''.join( str(predictions[1].astype(int)))
        # text_tensor = tf.make_tensor_proto(value, dtype=tf.string)
        # meta = tf.SummaryMetadata()
        # meta.plugin_data.plugin_name = "text"
        # summary.value.add(tag="whatever", metadata=meta, tensor=text_tensor)
        # variable_writer.add_summary(summary)

    summary = tf.Summary()

    summary.value.add(tag='loss_training', simple_value=loss)
    variable_writer.add_summary(summary, itN)
    # calculate the accuracy
    Batch_accuracy = 1.0 * accuracy(Batch_X, Batch_Y, predictions) / len(Batch_X)

    summary.value.add(tag='accuracy_training', simple_value=Batch_accuracy)
    variable_writer.add_summary(summary, itN)

    # link_weight_error
    link_weight_error = link_weight_error_percentage(Batch_Y, predictions)

    summary.value.add(tag='low_link_weight_error_training', simple_value=link_weight_error[0])
    variable_writer.add_summary(summary, itN)

    summary.value.add(tag='high_link_weight_error_training', simple_value=link_weight_error[1])
    variable_writer.add_summary(summary, itN)

    # error for 1, L, 2L
    link_weight_error, high_error, half_high_error = strauted_link_weight_error_percentage(Batch_Y,
                                                                                           Batch_Y_target,
                                                                                           predictions)
    summary.value.add(tag='link_weight_error', simple_value=link_weight_error)
    variable_writer.add_summary(summary, itN)
    # print('iterationN',iterationN,'score',score, '__',tf_metric_update)

    summary = tf.Summary()
    summary.value.add(tag='high_error', simple_value=high_error)
    variable_writer.add_summary(summary, itN)

    summary = tf.Summary()
    summary.value.add(tag='half_high_error', simple_value=half_high_error)
    variable_writer.add_summary(summary, itN)

def testWithIteration(sess, test_data, NN, variable_writer):
    X_test = test_data["X_test"]
    Y_test = test_data["Y_test"]
    # print('testing_data',testing_data)
    # Initialize session
    # merged = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
    # Test model
    if (FLAGS.rnn_cell):
        test_loss, test_predictions = sess.run([NN.loss, NN.y],
                                               {NN.x: X_test, NN.labels: Y_test, NN.batch_size: len(X_test),
                                                NN.is_train: True})
    else:
        test_loss, test_predictions = sess.run([NN.loss, NN.y], {NN.x: X_test, NN.labels: Y_test, NN.is_train: True})
    test_accuracy = 1.0 * accuracy(X_test, Y_test, test_predictions) / len(X_test)

    # save loss in tensorboard
    summary = tf.Summary()
    summary.value.add(tag='loss_test', simple_value=test_loss)
    variable_writer.add_summary(summary, 1)

    # save accuracy in tensorboard
    summary = tf.Summary()
    summary.value.add(tag='accuracy_test', simple_value=test_accuracy)
    variable_writer.add_summary(summary, 1)

    return test_loss, test_accuracy

def validationWithIteration(sess, valid_data, NN, variable_writer, it_epoch, nbBatch):
    # Test avec le validation test
    X_valid = valid_data["X_valid"]
    Y_valid = valid_data["Y_valid"]

    if (FLAGS.rnn_cell):
        valid_loss, valid_predictions = sess.run([NN.loss, NN.y],
                                                 {NN.x: X_valid, NN.labels: Y_valid, NN.batch_size: len(X_valid),
                                                  NN.is_train: True})
    else:
        valid_loss, valid_predictions = sess.run([NN.loss, NN.y],
                                                 {NN.x: X_valid, NN.labels: Y_valid, NN.is_train: True})

    valid_accuracy = (1.0 * accuracy(X_valid, Y_valid, valid_predictions)) / len(X_valid)

    # save loss in tensorboard
    summary = tf.Summary()
    summary.value.add(tag='loss_validation', simple_value=valid_loss)
    variable_writer.add_summary(summary, (it_epoch + 1) * nbBatch * FLAGS.batch_size)

    # save accuracy in tensorboard
    summary = tf.Summary()
    summary.value.add(tag='accuracy_validation', simple_value=valid_accuracy)
    variable_writer.add_summary(summary, (it_epoch + 1) * nbBatch * FLAGS.batch_size)

def accuracyLinkToPath(label, prediction):
    if len(label) != len(prediction):
        raise ("error label and prediction don't have the same length")
    it = 0.0


    for i in range(len(label)):

        if np.argmax(label[i]) == np.argmax(prediction[i]):
            it += 1.0



    return it / len(label)




