"""
Arrange training, testing and validation data to JSON file
"""

from Data_Set.Generation.Topology import Topology, traffic_s_d_to_index, Different_Topology
from Data_Set.Generation.CSP_algorithm import CPS, save_new_target_y,RWA_K_SPF_FF
# k_shortest_path_one_hot, \
#    k_shortest_path_multi_class
from Data_Set.Generation.YenKShortestPaths import YenKShortestPaths

from sklearn.model_selection import train_test_split
import pandas as pd
import json
import tensorflow as tf
import numpy as np
import pickle
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

def load_algoSolution(pathname, algoname):
    full_filename = pathname + algoname + ".json"
    file = open(full_filename, "r")
    sol = json.load(file)
    return sol  

def getDataComposition_RWA(data):
    '''
    :param data: label d'un jeu de donnée

    :return: chaîne de charactere décrivant le % de chaque classe dans :param data
    '''
    string = ""
    it = 0
    for i in range(len(data)):         
        if data[i][0] == 1:
            it += 1
    string += '\n' + 'no solution  = ' + str( it / (len(data))) + " %"
    
    for p in range(FLAGS.k_path_generated):
        for w in range(FLAGS.numWavelengths):
            pw_id = p*FLAGS.numWavelengths + w
            it = 0
            for i in range(len(data)):         
                if data[i][pw_id+1] == 1:
                    it += 1
            string += '\n' + 'path  ' +  str(p) + ' wavelength  ' +  str(w) + ' = ' + str( it / (len(data))) + " %"
            
    return  string

def datasetDescription(trainData, testData, validData, information, numLinks):
    '''
    :param trainData: training se
    :param testData: testing set
    :param validData: validation set
    :param information: liste des chemins vers les réseaux utilisés pour former se data set
    :param maxLinksLength: nombre maximum de liens dans l'ensemble des réseaux utilisés
    :return:
    '''
    string = ""
    string += '\n' + "datasets size : "
    string += '\n' + " training_data : "+ str(len(trainData[0]["X"]))+ " "+ str(len(trainData[0]["X"]) / FLAGS.nb_reqs)
    string += '\n' + " validation data : "+ str(len(validData["X"]))+ " "+ str(len(validData["X"]) / FLAGS.nb_reqs)
    string += '\n' + " testing_data : "+ str(len(testData["X"]))+ " "+ str(len(testData["X"]) / FLAGS.nb_reqs)
#    string += '\n' + " topology train: "+ str(len(trainData["topology"]))+ " "+ str(len(trainData["topology"]) / FLAGS.nb_reqs)
#    string += '\n' + " topology valid: "+ str(len(validData["topology"]))+ " "+ str(len(validData["topology"]) / FLAGS.nb_reqs)
#    string += '\n' + " topology test: "+ str(len(testData["topology"]))+ " "+ str(len(testData["topology"]) / FLAGS.nb_reqs)
    string += '\n' +  " numLinks " + str(numLinks)
    string += '\n\n\n' + "dataset used: "
    string += '\n' + information    
    for i in range(len(trainData)):
    #    string += '\n' + i
        string += '\n \n'  +"train DS composition i = " + str(i)
        string +=  getDataComposition_RWA(trainData[i]["y"])

    string += '\n \n'  +"validation DS composition"
    string +=  getDataComposition_RWA(validData["y"])

    string += '\n \n'  +"test DS composition"
    string +=  getDataComposition_RWA(testData["y"])

    return string
   
def generateDatasets(nb_dynTraces, traff_pathname, pathTopology, algo_name, newDS_folder, newDS_pathname, topology):
    # the raw DS is composed by a set of dyn traffic trace (requests) + netw states and its corresponding solutions
    # we classify the data traces as
    # - "nb_dynTraces-2" trining epoch
    # - 1 trace --> validation set to try after each epoch
    # - 1 trace --> test set to try after ALL the epochs
    nb_TrainEpochs = nb_dynTraces-2
    train = []
    for i in range(nb_dynTraces):
        dynTrace_pathname = traff_pathname + "trace_" + str(i) + "/"
        sol_pathname = pathTopology + dynTrace_pathname
        rawDS = load_algoSolution(sol_pathname, algo_name)

        print("i = ",i)
        print(len(train))
        #1. We remove the first 20% of the trace corresponding to the transient regime from a empty network state
        numReqs_transient = int(len(rawDS['X'])*0.2)    
#        print(len(rawDS['X']))
#        print(rawDS['X'][0])
        del rawDS['X'][0:numReqs_transient]
        del rawDS['y'][0:numReqs_transient]
#        print(len(rawDS['X']))
#        print(rawDS['X'][0])
    
        #2. Transforming input data(X) from raw represenations to (node/link) embedding representations                        
        if (FLAGS.state == 0):
            newDS = to_linkBased_dsRepres(rawDS, topology)
        else:
            newDS = to_nodeBased_dsRepres(rawDS, topology)  
        newDS['topology'] = topology["incMat"] 
            
        if i < nb_TrainEpochs :    
            train.append(newDS)
        elif i == nb_TrainEpochs :
            valid = newDS
        else :
            test = newDS
 #       x_data = newDS['X']
 #       y_data = newDS['y']
        
      
#    #np.random.seed(int(time.time()))
#    X_train_valid, X_test, y_train_valid, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state = 42)
#    test= {}
#    test['X'] = X_test
#    test['y'] = y_test
#    test['topology'] = topology["incMat"]
#    
#    #np.random.seed(int(time.time()))
#    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.25, random_state = 42)    
#    
#    valid= {}
#    valid['X'] = X_valid
#    valid['y'] = y_valid
#    valid['topology'] = topology["incMat"]
#    
#    train= {}
#    train['X'] = X_train
#    train['y'] = y_train
#    train['topology'] = topology["incMat"]
            
    #writing final datasets
    fullpathnme = newDS_folder + "state_" +str(FLAGS.state) + "/" + newDS_pathname 
    #+ "/trace_" + str(i) 
    if not os.path.exists(fullpathnme):
        os.makedirs(fullpathnme)  
#    full_filename = pathname + filename
#    if not os.path.exists(pathname):
#        os.makedirs(pathname)
#    with open(full_filename, 'w') as outfile:
#        print("Done : " , full_filename)
#        json.dump(list_LPmatrices, outfile)  
#        outfile.close()

    f = open(fullpathnme + algo_name + "_train.json", 'w+')
    print(type(train))
    pd.Series(train).to_json(path_or_buf=f)

    f = open(fullpathnme + algo_name + "_valid.json", 'w+')
    pd.Series(valid).to_json(path_or_buf=f)

    f = open(fullpathnme + algo_name + "_test.json", 'w+')
    pd.Series(test).to_json(path_or_buf=f)
    
    numLinks = len(topology["links"])
    dsi = datasetDescription(train, test, valid, newDS_pathname, numLinks)
    print(dsi)
    f = open(fullpathnme + algo_name + "_information.txt", 'w+')
    f.write(dsi)
                           
def generateDatasets_req2req(rawDS, topology, newDS_folder, newDS_pathname, algo_name):
    #Transforming input data(X) from raw represenations to (node/link) embedding representations                        
    if (FLAGS.state == 0):
        newDS = to_linkBased_dsRepres(rawDS, topology)
    else:
        # Codification avec l'état 2
        newDS = to_nodeBased_dsRepres(rawDS, topology)
        
    x_data = newDS['X']
    y_data = newDS['y']
    
    #np.random.seed(int(time.time()))
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state = 42)
    test= {}
    test['X'] = X_test
    test['y'] = y_test
    test['topology'] = topology["incMat"]
    
    #np.random.seed(int(time.time()))
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.25, random_state = 42)    
    
    valid= {}
    valid['X'] = X_valid
    valid['y'] = y_valid
    valid['topology'] = topology["incMat"]
    
    train= {}
    train['X'] = X_train
    train['y'] = y_train
    train['topology'] = topology["incMat"]
            
    #writing final datasets
    fullpathnme = newDS_folder + "state_" +str(FLAGS.state) + "/" + newDS_pathname
    if not os.path.exists(fullpathnme):
        os.makedirs(fullpathnme)  
#    full_filename = pathname + filename
#    if not os.path.exists(pathname):
#        os.makedirs(pathname)
#    with open(full_filename, 'w') as outfile:
#        print("Done : " , full_filename)
#        json.dump(list_LPmatrices, outfile)  
#        outfile.close()

    f = open(fullpathnme + algo_name + "_train.json", 'w+')
    pd.Series(train).to_json(path_or_buf=f)

    f = open(fullpathnme + algo_name + "_valid.json", 'w+')
    pd.Series(valid).to_json(path_or_buf=f)

    f = open(fullpathnme + algo_name + "_test.json", 'w+')
    pd.Series(test).to_json(path_or_buf=f)
    
    numLinks = len(topology["links"])
    dsi = datasetDescription(train, test, valid, newDS_pathname, numLinks)
    print(dsi)
    f = open(fullpathnme + algo_name + "_information.txt", 'w+')
    f.write(dsi)
    
    
def to_linkBased_dsRepres(dataset, topology):
    edges = topology["links"]
    numLinks = len(edges)
    node_vs_path_bin_table = np.array(topology['node_vs_path_bin_table'])    
    numNodes = node_vs_path_bin_table.shape[0]
    STATE_SIZE = numLinks*FLAGS.numWavelengths + numNodes
    
    newDataset = {}
    newDataset['X'] = []
    newDataset["y"] = dataset["y"]
    newDataset['topology'] = []

    for i in range(len(dataset["X"])):
        # change link load to one-hot vector with avaible link capacity
        new_state = np.zeros(STATE_SIZE)
        new_state[0:numLinks*FLAGS.numWavelengths] = dataset["X"][i][0:numLinks*FLAGS.numWavelengths]
#        for l in range(numLinks):
#            for w in range(FLAGS.numWavelengths):
#                lw_id = l*FLAGS.numWavelengths + w
#                new_state[lw_id] = dataset["X"][i][lw_id]
        ssnode = int(dataset["X"][i][len(edges)*FLAGS.numWavelengths])
        desdesnode = int(dataset["X"][i][len(edges)*FLAGS.numWavelengths + 1])
        volume = int(dataset["X"][i][len(edges)*FLAGS.numWavelengths + 2])
        new_state[numLinks * FLAGS.numWavelengths + ssnode] = -volume
        new_state[numLinks * FLAGS.numWavelengths + desdesnode] = volume
        
        newDataset["X"].append(list(new_state))
    
    return newDataset

def to_nodeBased_dsRepres(dataset, topology):
    
    edges = topology["links"]
    numLinks = len(edges)
    node_vs_path_bin_table = np.array(topology['node_vs_path_bin_table'])    
    numNodes = node_vs_path_bin_table.shape[0]
    STATE_SIZE = (1 + numLinks*FLAGS.numWavelengths) * numNodes
    
    newDataset = {}
    newDataset['X'] = []
    newDataset["y"] = dataset["y"]
    newDataset['topology'] = []
   
    for i in range(len(dataset["X"])):
        new_state = np.zeros(STATE_SIZE)
        for l in range(numLinks):
            left_node = edges[l][0]
            right_node = edges[l][1]
            for w in range(FLAGS.numWavelengths):
                lw_id = l*FLAGS.numWavelengths + w
                new_state[left_node * (numLinks*FLAGS.numWavelengths+ 1) + lw_id] = dataset["X"][i][lw_id]
                new_state[right_node * (numLinks*FLAGS.numWavelengths + 1) + lw_id] = dataset["X"][i][lw_id]
        ssnode = int(dataset["X"][i][len(edges)*FLAGS.numWavelengths])
        desdesnode = int(dataset["X"][i][len(edges)*FLAGS.numWavelengths + 1])
        volume = int(dataset["X"][i][len(edges)*FLAGS.numWavelengths + 2])
        new_state[(numLinks*FLAGS.numWavelengths+ 1) * (ssnode + 1) - 1] = -volume
        new_state[(numLinks*FLAGS.numWavelengths+ 1) * (desdesnode + 1) - 1] = volume

        newDataset["X"].append(list(new_state))
        
    return newDataset 