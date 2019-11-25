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
    string += '\n' + " training_data : "+ str(len(trainData["X"]))+ " "+ str(len(trainData["X"]) / FLAGS.nb_reqs)
    string += '\n' + " validation data : "+ str(len(validData["X"]))+ " "+ str(len(validData["X"]) / FLAGS.nb_reqs)
    string += '\n' + " testing_data : "+ str(len(testData["X"]))+ " "+ str(len(testData["X"]) / FLAGS.nb_reqs)
#    string += '\n' + " topology train: "+ str(len(trainData["topology"]))+ " "+ str(len(trainData["topology"]) / FLAGS.nb_reqs)
#    string += '\n' + " topology valid: "+ str(len(validData["topology"]))+ " "+ str(len(validData["topology"]) / FLAGS.nb_reqs)
#    string += '\n' + " topology test: "+ str(len(testData["topology"]))+ " "+ str(len(testData["topology"]) / FLAGS.nb_reqs)
    string += '\n' +  " numLinks " + str(numLinks)
    string += '\n\n\n' + "dataset used: "
    string += '\n' + information    
    #for i in information:
    #    string += '\n' + i
    string += '\n \n'  +"train DS composition"
    string +=  getDataComposition_RWA(trainData["y"])

    string += '\n \n'  +"validation DS composition"
    string +=  getDataComposition_RWA(validData["y"])

    string += '\n \n'  +"test DS composition"
    string +=  getDataComposition_RWA(testData["y"])

    return string
                              
def generateDatasets(rawDS, topology, newDS_folder, newDS_pathname, algo_name):
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
#
## return traffic instances from file
#def Traffic(Folderpath, filename):
#    # TODO: add networks_name to a string in the folder_path
#    #Folderpath = "Data_Set/" + FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "node-" + str(degree) + "degree/"
#    #filename = str(n) + "_links.txt"
#    #file = open(Folderpath + filename, "r")
#
#    file = open(Folderpath + filename, "r")
#    #file = open("generatedGraphsData/" + str(FLAGS.network_node) + "-node/" +"uniformtraffic.txt", "r")
#    traffic_matrix = []
#    for line in file.readlines():
#        traffic_matrix.append([int(x) for x in line.split(' ')])
#    return traffic_matrix
#
#def generateSample(Folderpath, synthetictraffic):
#    
#    # A sample is an instance of a set of LP requests
#    
#    G = Different_Topology(Folderpath)
#    if synthetictraffic:
#        trafficfilename = "synthetictraffic.txt"
#    else:
#        trafficfilename = "uniformtraffic.txt"
#    traffic_matrix = Traffic(Folderpath, trafficfilename)
#    print('traffic_matrix done')
#    
#    # defining path set
#    sd_vs_path_bin_table, link_vs_path_bin_table, node_vs_path_bin_table = computingPathSet(G, Folderpath)
#    
#    # trivail RWA solution with heur SPF-FF
#    
#    
#       #[x_data, y_data] contains "Total_R" samples (as many as commodities "c" in "traffic_matrix"), e.g, 10000
#    # We have a "traffic_matrix" (synthetic or uniform) per (number of nodes, avg degree, random instance)
#    x_data, y_data = RWA_K_SPF_FF(G, traffic_matrix, Folderpath, percentage_of_congestion, numLinks, FLAGS.numWavelengths)
#    
#def computingPathSet(G, Folderpath):   
#    GGG = G.copy()
#    numLinks = GGG.number_of_edges()
#    numNodes = len(GGG)
#    for i in range(numLinks):
#        GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = 1.0
#
#    yksp = YenKShortestPaths(GGG)
#    matrix_shortest_path = []
#    sd_vs_path_bin_table = np.zeros((numNodes*numNodes, FLAGS.k_path_generated))
#    link_vs_path_bin_table = np.zeros((numLinks, FLAGS.k_path_generated))
#    node_vs_path_bin_table = np.zeros((numNodes, FLAGS.k_path_generated))
#    
#    # Generation of the matrix with the paths
#    path = 0
#    for i in range(numNodes) :
#        for j in range(numNodes):
#            tmpLine = []
#            sd = i*numNodes + j
#            if (i != j):
#                if yksp.findFirstShortestPath(i,j) == None :
#                    print('i',i,'j',j, ' are not connected')
#                    continue
#                sPath = yksp.findFirstShortestPath(i,j).nodeList
#                tmpLine.append(sPath)
#                sd_vs_path_bin_table[sd][path] = 1
#                for n in range(len(sPath) - 1): #i source nodes in path
#                    node_vs_path_bin_table[n][path] = 1
#                    node_vs_path_bin_table[n+1][path] = 1
#                    edge = (sPath[n], sPath[n + 1])
#                    if edge in GGG.edges():
#                        edge_index = GGG.edges().index(edge)
#                        link_vs_path_bin_table[edge_index][path] = 1
#                path += 1
#                # for each input we want FLAGS.k_path_generated labels
#                #print('sp',yksp.findFirstShortestPath(i,j).nodeList)
#                for k in range(FLAGS.k_path_generated - 1 ):
#                    kPATH = yksp.getNextShortestPath()
#                    if (kPATH == None):
#                        #print("is None")
#                        if(k==0):
#                            tmpLine.append(tmpLine[0])
#                        else:
#                            tmpLine.append(tmpLine[k])
#                    else:
#                        tmpLine.append(kPATH.nodeList)
#                        sd_vs_path_bin_table[sd][path] = 1
#                        for n in range(len(kPATH) - 1): #i source nodes in path
#                            node_vs_path_bin_table[n][path] = 1
#                            node_vs_path_bin_table[n+1][path] = 1
#                            edge = (kPATH[n], kPATH[n + 1])
#                            if edge in GGG.edges():
#                                edge_index = GGG.edges().index(edge)
#                                link_vs_path_bin_table[edge_index][path] = 1
#                                path += 1
#                    #print(k,yksp.getNextShortestPath())
#                    #print(yksp.getNextShortestPath().nodeList)
#                    #if(yksp.getNextShortestPath() == None):
#                    #    print("is None")
#                    #tmpLine.append(yksp.getNextShortestPath().nodeList)
#                    #print('tmpLine',tmpLine)
#                matrix_shortest_path.append(tmpLine)
#
#    with open( Folderpath + "matriceShortestPath_RWA.txt", 'wb') as f:
#        if not os.path.exists(Folderpath + "matriceShortestPath_RWA.txt"):
#            os.makedirs(Folderpath + "matriceShortestPath_RWA.txt")
#    #with open("Data_Set/" + FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/matriceShortestPath.txt", 'wb') as f:
#        pickle.dump(matrix_shortest_path, f)
#        
#    return sd_vs_path_bin_table, link_vs_path_bin_table, node_vs_path_bin_table         
#
#
#    
#def generate_dataset_linkToWeight():
#    """
#    generate training, testing and validation data
#    :return:
#    """
#    G = Topology()
#    traffic_matrix = Traffic()
#    x_data, y_data = CPS(G, traffic_matrix)
#    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 42)
#    train = {}
#    train['X_train'] = []
#    train['Y_train'] = []
#    for i in range(len(X_train)):
#        train['X_train'].append(list(X_train[i]))
#        train['Y_train'].append(list(y_train[i]))
#
#    test = {}
#    test['X_test'] = []
#    test['Y_test'] = []
#    for i in range(len(X_test)):
#        test['X_test'].append(list(X_test[i]))
#        test['Y_test'].append(list(y_test[i]))
#
#    generatedGraphsPath = "Data_Set/" +  FLAGS.dataset_name  + "/"
#    graphName = FLAGS.network_node
#    graphPath = generatedGraphsPath + str(graphName) +  "-node/"
#
#    with open( graphPath + str(graphName) + "-node-training.json", 'w') as outfile:
#        json.dump(train, outfile)
#
#    with open( graphPath + str(graphName) + "-node-testing.json", 'w') as outfile:
#        json.dump(test, outfile)
#
#    save_new_target_y(train['X_train'], test['X_test'])
#    return 0
#
#
#def generate_dataset_linkToPath(Folderpath, percentage_of_congestion, numLinks, synthetictraffic):
#    """
#    Degree = [2, 4, 8]
#    for i in range(len(Degree)):
#        degree = Degree[i]
#        for n in range(FLAGS.nb_random_graph):
#            G = Different_Topology(degree, n)
#    """
#    G = Different_Topology(Folderpath)
#    if synthetictraffic:
#        trafficfilename = "synthetictraffic.txt"
#    else:
#        trafficfilename = "uniformtraffic.txt"
#    traffic_matrix = Traffic(Folderpath, trafficfilename)
#    print('traffic_matrix done')
#
#
#    #[x_data, y_data] contains "Total_R" samples (as many as commodities "c" in "traffic_matrix"), e.g, 10000
#    # We have a "traffic_matrix" (synthetic or uniform) per (number of nodes, avg degree, random instance)
#    x_data, y_data = RWA_K_SPF_FF(G, traffic_matrix, Folderpath, percentage_of_congestion, numLinks, FLAGS.numWavelengths)
#    #x_data, y_data = k_shortest_path_one_hot(G, traffic_matrix, Folderpath, percentage_of_congestion, numLinks)
#
#    #print('k_shortest_path_multi_class done')
#    #print('x_data',x_data)
#    #print('y_data', y_data)
#    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
#    train = {}
#    train['X_train'] = []
#    train['Y_train'] = []
#    for i in range(len(X_train)):
#        train['X_train'].append(list(X_train[i]))
#        train['Y_train'].append(list(y_train[i]))
#
#    test = {}
#    test['X_test'] = []
#    test['Y_test'] = []
#    for i in range(len(X_test)):
#        test['X_test'].append(list(X_test[i]))
#        test['Y_test'].append(list(y_test[i]))
#
#    if(synthetictraffic):
#        graphName = str(percentage_of_congestion) + "_synthetictraffic"
#    else:
#        graphName = str(percentage_of_congestion) + "_uniformtraffic"
#
#    # with open(graphPath + str(graphName) + "-node-training.json", 'w') as outfile:
#    with open(Folderpath + graphName + "-node-training.json", 'w') as outfile:
#        print("Done : " , Folderpath + graphName + "-node-training.json")
#        json.dump(train, outfile)
#
#    # with open(graphPath + str(graphName) + "-node-testing.json", 'w') as outfile:
#    with open(Folderpath + graphName + "-node-testing.json", 'w') as outfile:
#        json.dump(test, outfile)
#
#    #save_new_target_y(train['X_train'], test['X_test'], Folderpath)
#
#    return 0
#
#
#def K_shortest_path_s_d(s,d):
#    K_shortestpath = K_shortest_path()
#    print('s', s, 'd', d, K_shortestpath[traffic_s_d_to_index(s, d)])
#
## get all k shortest path for all s and d
#def K_shortest_path():
#    G = Topology()
#    for i in range(FLAGS.network_link):
#        G[G.edges()[i][0]][G.edges()[i][1]]['weight'] = 1
#
#
#    K_shortestpath = [[] for i in range(FLAGS.network_node*FLAGS.network_node-FLAGS.network_node)]
#
#    print(K_shortestpath)
#    print(K_shortestpath[0])
#    ShortPaths = YenKShortestPaths(G)
#    index = -1
#    for i in range(FLAGS.network_node):
#        for j in range(FLAGS.network_node):
#            if(i!=j):
#                index +=1
#                path0 = (ShortPaths.findFirstShortestPath(i, j)).nodeList
#                path1 = ShortPaths.getNextShortestPath().nodeList
#                path2 = ShortPaths.getNextShortestPath().nodeList
#                K_shortestpath[index].append(path0)
#                K_shortestpath[index].append(path1)
#                K_shortestpath[index].append(path2)
#            #K_shortestpath[i][j][1] = path1
#            #K_shortestpath[i][j][2] = path2
#    return K_shortestpath
#
#
#
#def data_with_one_hot_state_edge_embedding(STATE_SIZE, training_data, testing_data):
#    train = training_data.copy()
#    test = testing_data.copy()
#    oldDict_X_training_data = train.copy()
#    del train["X_train"]
#    train["X_train"] = []
#    oldDict_X_test_data = test.copy()
#    del test["X_test"]
#
#    test["X_test"] = []
#    G = Topology()
#    for i in range(len(oldDict_X_training_data["X_train"])):
#    # change link load to one-hot vector with avaible link capacity
#        new_state = np.zeros(STATE_SIZE)
#        for link in range(FLAGS.network_link):
#            left_node = G.edges()[link][0]
#            right_node = G.edges()[link][1]
#            new_state[2 * FLAGS.network_node * link + left_node] = oldDict_X_training_data["X_train"][i][link]
#            new_state[2 * FLAGS.network_node * link + FLAGS.network_node + right_node] = oldDict_X_training_data["X_train"][i][link]
#        ssnode = int(oldDict_X_training_data["X_train"][i][FLAGS.network_link])
#        desdesnode = int(oldDict_X_training_data["X_train"][i][FLAGS.network_link + 1])
#        volume = int(oldDict_X_training_data["X_train"][i][FLAGS.network_link + 2])
#        '''
#        if volume == 1:
#            volume_index = 0
#        elif volume == 2:
#            volume_index = 1
#        elif volume == 8:
#            volume_index = 2
#        elif volume == 32:
#            volume_index = 3
#        elif volume == 64:
#            volume_index = 4
#        '''
#        new_state[FLAGS.network_link * 2 * FLAGS.network_node + ssnode] = volume
#        new_state[FLAGS.network_link * 2 * FLAGS.network_node + FLAGS.network_node + desdesnode] = volume
#
#        train["X_train"].append(list(new_state))
#
#
## print(' odl,', oldDict_X_training_data["X_train"][0],'new_state', new_state,
##      'X_train', training_data["X_train"][0])
#
#    for i in range(len(oldDict_X_test_data["X_test"])):
#    # change link load to one-hot vector with avaible link capacity
#        new_state = np.zeros(STATE_SIZE)
#        for link in range(FLAGS.network_link):
#            left_node = G.edges()[link][0]
#            right_node = G.edges()[link][1]
#            new_state[2 * FLAGS.network_node * link + left_node] = oldDict_X_test_data["X_test"][i][link]
#            new_state[2 * FLAGS.network_node * link + FLAGS.network_node + right_node] = \
#                oldDict_X_test_data["X_test"][i][link]
#        ssnode = int(oldDict_X_test_data["X_test"][i][FLAGS.network_link])
#        desdesnode = int(oldDict_X_test_data["X_test"][i][FLAGS.network_link + 1])
#        volume = int(oldDict_X_test_data["X_test"][i][FLAGS.network_link + 2])
#        '''
#        if volume == 1:
#            volume_index = 0
#        elif volume == 2:
#            volume_index = 1
#        elif volume == 8:
#            volume_index = 2
#        elif volume == 32:
#            volume_index = 3
#        elif volume == 64:
#            volume_index = 4
#        '''
#        new_state[FLAGS.network_link * 2 * FLAGS.network_node + ssnode] = volume
#        new_state[FLAGS.network_link * 2 * FLAGS.network_node + FLAGS.network_node + desdesnode] = volume
#
#        test["X_test"].append(list(new_state))
#    return train, test
#
#
#def data_with_one_hot_state_node_embedding(STATE_SIZE, training_data, testing_data):
#    train = training_data.copy()
#    test = testing_data.copy()
#    oldDict_X_training_data = train.copy()
#    del train["X_train"]
#    train["X_train"] = []
#    oldDict_X_test_data = test.copy()
#    del test["X_test"]
#
#    test["X_test"] = []
#    G = Topology()
#    for i in range(len(oldDict_X_training_data["X_train"])):
#
#    # change link load to one-hot vector with avaible link capacity
#        new_state = np.zeros(STATE_SIZE)
#        for link in range(FLAGS.network_link):
#            left_node = G.edges()[link][0]
#            right_node = G.edges()[link][1]
#            new_state[left_node * (FLAGS.network_link + 1) + link] = -oldDict_X_training_data["X_train"][i][link]
#            new_state[right_node * (FLAGS.network_link + 1) + link] = oldDict_X_training_data["X_train"][i][link]
#        ssnode = int(oldDict_X_training_data["X_train"][i][FLAGS.network_link])
#        desdesnode = int(oldDict_X_training_data["X_train"][i][FLAGS.network_link + 1])
#        volume = int(oldDict_X_training_data["X_train"][i][FLAGS.network_link + 2])
#
#        new_state[(FLAGS.network_link+1) * (ssnode+1) - 1] = -volume
#        new_state[(FLAGS.network_link+1) * (desdesnode+1) - 1] = volume
#        train["X_train"].append(list(new_state))
#
#
#    #print(' odl,', oldDict_X_training_data["X_train"][0], 'X_train', train["X_train"][0])
#
#    for i in range(len(oldDict_X_test_data["X_test"])):
#    # change link load to one-hot vector with avaible link capacity
#        new_state = np.zeros(STATE_SIZE)
#        for link in range(FLAGS.network_link):
#            left_node = G.edges()[link][0]
#            right_node = G.edges()[link][1]
#            new_state[left_node * (FLAGS.network_link + 1) + link] = -oldDict_X_test_data["X_test"][i][link]
#            new_state[right_node * (FLAGS.network_link + 1) + link] = oldDict_X_test_data["X_test"][i][link]
#        ssnode = int(oldDict_X_test_data["X_test"][i][FLAGS.network_link])
#        desdesnode = int(oldDict_X_test_data["X_test"][i][FLAGS.network_link + 1])
#        volume = int(oldDict_X_test_data["X_test"][i][FLAGS.network_link + 2])
#        new_state[(FLAGS.network_link+1) * (ssnode+1) - 1] = -volume
#        new_state[(FLAGS.network_link+1) * (desdesnode+1) - 1] = volume
#
#        test["X_test"].append(list(new_state))
#    return train, test
#
#def data_with_one_hot_state_node_embedding(STATE_SIZE, training_data, testing_data):
#    train = training_data.copy()
#    test = testing_data.copy()
#    oldDict_X_training_data = train.copy()
#    del train["X_train"]
#    train["X_train"] = []
#    oldDict_X_test_data = test.copy()
#    del test["X_test"]
#
#    test["X_test"] = []
#    G = Topology()
#    for i in range(len(oldDict_X_training_data["X_train"])):
#    # change link load to one-hot vector with avaible link capacity
#        new_state = np.zeros(STATE_SIZE)
#        for link in range(FLAGS.network_link):
#            left_node = G.edges()[link][0]
#            right_node = G.edges()[link][1]
#            new_state[left_node * (FLAGS.network_link + 1) + link] = -oldDict_X_training_data["X_train"][i][link]
#            new_state[right_node * (FLAGS.network_link + 1) + link] = oldDict_X_training_data["X_train"][i][link]
#        ssnode = int(oldDict_X_training_data["X_train"][i][FLAGS.network_link])
#        desdesnode = int(oldDict_X_training_data["X_train"][i][FLAGS.network_link + 1])
#        volume = int(oldDict_X_training_data["X_train"][i][FLAGS.network_link + 2])
#
#        new_state[(FLAGS.network_link+1) * (ssnode+1) - 1] = -volume
#        new_state[(FLAGS.network_link+1) * (desdesnode+1) - 1] = volume
#        train["X_train"].append(list(new_state))
#
#
#    #print(' odl,', oldDict_X_training_data["X_train"][0], 'X_train', train["X_train"][0])
#
#    for i in range(len(oldDict_X_test_data["X_test"])):
#    # change link load to one-hot vector with avaible link capacity
#        new_state = np.zeros(STATE_SIZE)
#        for link in range(FLAGS.network_link):
#            left_node = G.edges()[link][0]
#            right_node = G.edges()[link][1]
#            new_state[left_node * (FLAGS.network_link + 1) + link] = -oldDict_X_test_data["X_test"][i][link]
#            new_state[right_node * (FLAGS.network_link + 1) + link] = oldDict_X_test_data["X_test"][i][link]
#        ssnode = int(oldDict_X_test_data["X_test"][i][FLAGS.network_link])
#        desdesnode = int(oldDict_X_test_data["X_test"][i][FLAGS.network_link + 1])
#        volume = int(oldDict_X_test_data["X_test"][i][FLAGS.network_link + 2])
#        new_state[(FLAGS.network_link+1) * (ssnode+1) - 1] = -volume
#        new_state[(FLAGS.network_link+1) * (desdesnode+1) - 1] = volume
#
#        test["X_test"].append(list(new_state))
#    return train, test