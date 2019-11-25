# coding=utf-8
"""
CPS algorithm for traffic routing to generate the target value (the routing weight)
"""

import json
import pickle
import numpy as np
import tensorflow as tf
import os
from Data_Set.Generation.Topology import Topology, traffic_s_d_to_index
from Data_Set.Generation.YenKShortestPaths import YenKShortestPaths

flags = tf.app.flags
FLAGS = flags.FLAGS

NUM_REQUEST_per_solution = 400

def CPS(G,Traffic_Matrix):
    """
    :param G:
    :param Traffic_Matrix:
    :return: target value for y
       link weight: 1            for links on shortest path,
                   NUM_EDGES * 2 for other links
                    0            for all the links if there does not exist a routing path
    """
    CSP_avg_bd = 0
    CSP_num_staturations = 0
    CSP_link_load = np.zeros(numLinks)

    C = len(Traffic_Matrix)

    y_target_data = np.zeros((C, numLinks))
    y_target_data_path = np.zeros((C, numLinks))

    # two types for x_data:
    # 1: [link_load,s,d,volume]
    x_data = np.zeros((C, numLinks+3))
    # 2: two vector indicating left and right nodes, three one-hot vector for s, d, and volume
    #x_data[c] = np.zeros((C, NUM_EDGES*NUM_NODES*2+2*NUM_NODES+NUM_SERVICES))

    for c in range(C):
        if(c % NUM_REQUEST_per_solution == 0):
            CSP_link_load = np.zeros(numLinks)
        source = Traffic_Matrix[c][0]
        destination = Traffic_Matrix[c][1]
        bandwidth = Traffic_Matrix[c][2]
        for i in range(numLinks):
            x_data[c][i] = FLAGS.link_capacity - CSP_link_load[i]  #avaiable link load
            if (x_data[c][i] >= bandwidth):
                y_target_data[c][i] = 1
            else:
                y_target_data[c][i] = numLinks*2
        x_data[c][numLinks] = source
        x_data[c][numLinks + 1] = destination
        x_data[c][numLinks + 2] = bandwidth

        #print('c',c,'x_data',x_data[c])

        #print('c',source,destination,bandwidth)
        GGG = G.copy()
        for i in range(numLinks):
            #print('i',i,'G.edges()[i]',G.edges()[i])
            if (CSP_link_load[i] + bandwidth > FLAGS.link_capacity):
                GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = numLinks*2
            else:
                GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = 1

        CSP_Path = YenKShortestPaths(GGG)
        CSP_Fpath = (CSP_Path.findFirstShortestPath(source, destination)).nodeList
        #print(CSP_Fpath)
        #todo:  should be less than numLinks ** 2
        CSP_saturation = False
        for i in range(len(CSP_Fpath) - 1):
            edge = (CSP_Fpath[i], CSP_Fpath[i + 1])
            if edge in GGG.edges():
                edge_index = GGG.edges().index(edge)
                if (bandwidth + CSP_link_load[edge_index] > GGG[CSP_Fpath[i]][CSP_Fpath[i + 1]]['capacity']):
                    CSP_saturation = True
                    break

        if (CSP_saturation):
            CSP_num_staturations += 1
        else:
            CSP_avg_bd = CSP_avg_bd + bandwidth
            for a in range(numLinks):
                y_target_data_path[c][a] = numLinks*2
            for i in range(len(CSP_Fpath) - 1):
                edge = (CSP_Fpath[i], CSP_Fpath[i + 1])
                if edge in GGG.edges():
                    edge_index = GGG.edges().index(edge)
                    y_target_data_path[c][edge_index] = 1
                    CSP_link_load[edge_index] = CSP_link_load[edge_index] + bandwidth
        #print(y_target_data[c])


    #TODO: save the target value to trainning instances

    return x_data, y_target_data

def CPS_get_new_taget(G,X_data):
    '''
    :param G:
    :param X_data:
    :return:  y_target_data, [1,1,L,2L],
    1 indicates the path,
    L indicates the link not on the path and not saturated link
    2L indicates the saturated link
    '''
    #C = FLAGS.max_total_steps# len(X_data)
    C =  len(X_data)
    y_target_data = np.ones((C, numLinks))

    for c in range(C):
        source = X_data[c][numLinks]
        destination = X_data[c][numLinks+1]
        bandwidth = X_data[c][numLinks+2]
        GGG = G.copy()
        for i in range(numLinks):
            if (bandwidth > X_data[c][i] ):
                y_target_data[c][i] = numLinks*2
                GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = numLinks * 2
            else:
                y_target_data[c][i] = numLinks
                GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = 1

        CSP_Path = YenKShortestPaths(GGG)
        CSP_Fpath = (CSP_Path.findFirstShortestPath(source, destination)).nodeList
        CSP_saturation = False
        for i in range(len(CSP_Fpath) - 1):
            edge = (CSP_Fpath[i], CSP_Fpath[i + 1])
            if edge in GGG.edges():
                edge_index = GGG.edges().index(edge)
                if (bandwidth > X_data[c][edge_index]):
                    CSP_saturation = True
                    break
        if (not CSP_saturation):
            for i in range(len(CSP_Fpath) - 1):
                edge = (CSP_Fpath[i], CSP_Fpath[i + 1])
                if edge in GGG.edges():
                    edge_index = GGG.edges().index(edge)
                    y_target_data[c][edge_index] = 1
        #print('c', c)
        #print('X_data', X_data[c])
        #print('y_target_data', y_target_data[c])

    return y_target_data

def save_new_target_y(X_train_data, X_test_data):
    G = Topology()
    print('X_train_data',len(X_train_data),'X_test_data',len(X_test_data))
    Y_train_new_target = CPS_get_new_taget(G,X_train_data)
    Y_test_new_target = CPS_get_new_taget(G,X_test_data)
    y_new_target_data = {}
    y_new_target_data['Y_new_train'] = []
    y_new_target_data['Y_new_test'] = []
    for i in range(len(Y_train_new_target)):
        y_new_target_data['Y_new_train'].append(list(Y_train_new_target[i]))
    for i in range(len(Y_test_new_target)):
        y_new_target_data['Y_new_test'].append(list(Y_test_new_target[i]))

    generatedGraphsPath = "Data_Set/generatedGraphsData/"
    graphName = FLAGS.network_node
    graphPath = generatedGraphsPath + str(graphName) +  "-node/"
    with open(graphPath + str(graphName) + "-node-Y_new_target.json", 'w') as outfile:
        json.dump(y_new_target_data, outfile)

    return 0

def link_wavelength_id(link_id, wavelength_id, numWavelengths):
    return link_id*numWavelengths + wavelength_id
# the soltions are Shortest Path First (among the K shortest paths) for the
# routing and First Fit for the wavlenghth assignment 
def RWA_K_SPF_FF(G, Traffic_Matrix, Folderpath, percentage_of_congestion, numLinks, numWavelengths):
    nb_wavelength_blockings = 0
    nb_current_request = 0
    link_wavelength_state = np.zeros(numLinks*numWavelengths) # 1 if (l,w) is used

    C = len(Traffic_Matrix)
#    print(FLAGS.k_path_generated)
#    print(numWavelengths)
#    print(FLAGS.k_path_generated*numWavelengths)
    y = np.zeros((C, FLAGS.k_path_generated*numWavelengths + 1)) # first position [0] is no_solution
    x_data = np.zeros((C, numLinks*numWavelengths + 3))

    GGG = G.copy()
    for i in range(numLinks):
        GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = 1.0


    yksp = YenKShortestPaths(GGG)
    matrix_shortest_path = []

    '''
    sp_path = yksp.findFirstShortestPath(0, 1).nodeList
    print('sp_path',sp_path)
    two_sp_path = yksp.getNextShortestPath().nodeList
    print('two_sp_path', two_sp_path)
    thrid_sp_path = yksp.getNextShortestPath().nodeList
    print('thrid_sp_path', thrid_sp_path)
    '''
    # Generation of the matrix with the paths
    for i in range(FLAGS.network_node) :

        for j in range(FLAGS.network_node ):
            tmpLine = []
            if (i != j):
                if yksp.findFirstShortestPath(i,j) == None :
                    print('i',i,'j',j, ' are not connected')
                    continue
                tmpLine.append(yksp.findFirstShortestPath(i,j).nodeList)
                # for each input we want FLAGS.k_path_generated labels
                #print('sp',yksp.findFirstShortestPath(i,j).nodeList)
                for k in range(FLAGS.k_path_generated - 1 ):
                    kPATH = yksp.getNextShortestPath()
                    if (kPATH == None):
                        #print("is None")
                        if(k==0):
                            tmpLine.append(tmpLine[0])
                        else:
                            tmpLine.append(tmpLine[k])
                    else:
                        tmpLine.append(kPATH.nodeList)
                    #print(k,yksp.getNextShortestPath())
                    #print(yksp.getNextShortestPath().nodeList)
                    #if(yksp.getNextShortestPath() == None):
                    #    print("is None")
                    #tmpLine.append(yksp.getNextShortestPath().nodeList)
                    #print('tmpLine',tmpLine)
                matrix_shortest_path.append(tmpLine)

    with open( Folderpath + "matriceShortestPath_RWA.txt", 'wb') as f:
        if not os.path.exists(Folderpath + "matriceShortestPath_RWA.txt"):
            os.makedirs(Folderpath + "matriceShortestPath_RWA.txt")
    #with open("Data_Set/" + FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/matriceShortestPath.txt", 'wb') as f:
        pickle.dump(matrix_shortest_path, f)


    #matrix_shortest_path.dump(FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/matriceShortestPath.txt")
    #np.savetxt(FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/matriceShortestPath.txt", matrix_shortest_path , fmt='%d')
    #np.savez(FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/matriceShortestPath.txt", matrix_shortest_path )

    for c in range(C):
        nb_current_request += 1
        # when we have more than FLAGS.percentage_of_congestion of congestion in our exemple
        # we reset the capacity of each link
        if ( nb_wavelength_blockings / nb_current_request > percentage_of_congestion):
            print('nb_wavelength_blockings', nb_wavelength_blockings)
            print('nb_current_request', nb_current_request)
            print('con ', nb_wavelength_blockings / nb_current_request)
            print('percentage_of_congestion', percentage_of_congestion)
            # print("reset à ", nb_staturations / nb_current_request)
            link_wavelength_state = np.zeros(numLinks*numWavelengths)
            nb_wavelength_blockings = 0
            nb_current_request = 0

        # input computation
        source = Traffic_Matrix[c][0]
        destination = Traffic_Matrix[c][1]
        bandwidth = Traffic_Matrix[c][2]
        for i in range(numLinks):
            for w in range(numWavelengths):
                lw_id = link_wavelength_id(i, w, numWavelengths)
                x_data[c][lw_id] = 1 - link_wavelength_state[lw_id]
        x_data[c][numLinks*numWavelengths] = source
        x_data[c][numLinks*numWavelengths + 1] = destination
        x_data[c][numLinks*numWavelengths + 2] = bandwidth


        # computation of the shortest path
        pendingBandwidthReq = bandwidth
        local_blocked_wavelengths = 0
        optimalPath = []
        for w in range(numWavelengths):
            GGG = G.copy()
            for i in range(numLinks):
                lw_id = link_wavelength_id(i, w, numWavelengths)
                if (link_wavelength_state[lw_id]  > 0):
                    GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = numLinks * 2
                else:
                    GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = 1
            CSP_Path = YenKShortestPaths(GGG)   #WARNING that can contain an infeasible path with link wights = numLinks * 2
            SPtmp = CSP_Path.findFirstShortestPath(source, destination).nodeList
            #Check if SPtmp is really feasible
            blocking = False
            for n in range(len(SPtmp) - 1): #i source nodes in path
                edge = (SPtmp[n], SPtmp[n + 1])
                if edge in GGG.edges():
                    edge_index = GGG.edges().index(edge)
                    tmp_lw_id = link_wavelength_id(edge_index, w, numWavelengths)
                    if (link_wavelength_state[tmp_lw_id] > 0):
                        blocking = True
                        break    
            if blocking: # we move to the next wavelength 
                local_blocked_wavelengths += 1
                continue
            else : # we stop searching a free wavelngth and we save the path
                for p in range(FLAGS.k_path_generated ):
                    if SPtmp == matrix_shortest_path[traffic_s_d_to_index(source,destination)][p] :
                        wav_path_id = link_wavelength_id(p, w, numWavelengths)
                        # we update the network status
                        y[c][wav_path_id+1] = 1
                        optimalPath = SPtmp
                        break
                break
            GGG.clear()        
        if (local_blocked_wavelengths == numWavelengths) : #no path available
            nb_wavelength_blockings  += 1
            y[c][0] = 1
            #print('blocking', blocking)
            #print('nb_wavelength_blockings', nb_wavelength_blockings)
        else: # we update the network status
            for i in range(len(optimalPath) - 1):
                edge = (optimalPath[i], optimalPath[i + 1])
                if edge in GGG.edges():
                    edge_index = GGG.edges().index(edge)
                    link_wavelength_state[edge_index] = link_wavelength_state[edge_index] + 1  

    return x_data,  y
    
    
def k_shortest_path_one_hot(G, Traffic_Matrix, Folderpath, percentage_of_congestion, numLinks):
    CSP_avg_bd = 0
    nb_staturations = 0
    nb_current_request = 0
    CSP_link_load = np.zeros(numLinks)

    C = len(Traffic_Matrix)
    y = np.zeros((C, FLAGS.k_path_generated + 1))
    x_data = np.zeros((C, numLinks + 3))

    GGG = G.copy()
    for i in range(numLinks):
        GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = 1.0


    yksp = YenKShortestPaths(GGG)
    matrix_shortest_path = []

    '''
    sp_path = yksp.findFirstShortestPath(0, 1).nodeList
    print('sp_path',sp_path)
    two_sp_path = yksp.getNextShortestPath().nodeList
    print('two_sp_path', two_sp_path)
    thrid_sp_path = yksp.getNextShortestPath().nodeList
    print('thrid_sp_path', thrid_sp_path)
    '''
    # Generation of the matrix with the paths
    for i in range(FLAGS.network_node) :

        for j in range(FLAGS.network_node ):
            tmpLine = []
            if (i != j):
                if yksp.findFirstShortestPath(i,j) == None :
                    print('i',i,'j',j, ' are not connected')
                    continue
                tmpLine.append(yksp.findFirstShortestPath(i,j).nodeList)
                # for each input we want FLAGS.k_path_generated labels
                #print('sp',yksp.findFirstShortestPath(i,j).nodeList)
                for k in range(FLAGS.k_path_generated - 1 ):
                    kPATH = yksp.getNextShortestPath()
                    if (kPATH == None):
                        #print("is None")
                        if(k==0):
                            tmpLine.append(tmpLine[0])
                        else:
                            tmpLine.append(tmpLine[k])
                    else:
                        tmpLine.append(kPATH.nodeList)
                    #print(k,yksp.getNextShortestPath())
                    #print(yksp.getNextShortestPath().nodeList)
                    #if(yksp.getNextShortestPath() == None):
                    #    print("is None")
                    #tmpLine.append(yksp.getNextShortestPath().nodeList)
                    #print('tmpLine',tmpLine)
                matrix_shortest_path.append(tmpLine)

    with open( Folderpath + "matriceShortestPath.txt", 'wb') as f:
        if not os.path.exists(Folderpath + "matriceShortestPath.txt"):
            os.makedirs(Folderpath + "matriceShortestPath.txt")
    #with open("Data_Set/" + FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/matriceShortestPath.txt", 'wb') as f:
        pickle.dump(matrix_shortest_path, f)


    #matrix_shortest_path.dump(FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/matriceShortestPath.txt")
    #np.savetxt(FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/matriceShortestPath.txt", matrix_shortest_path , fmt='%d')
    #np.savez(FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/matriceShortestPath.txt", matrix_shortest_path )

    for c in range(C):
        nb_current_request += 1
        # when we have more than FLAGS.percentage_of_congestion of congestion in our exemple
        # we reset the capacity of each link
        if ( nb_staturations / nb_current_request > percentage_of_congestion):
            # print("reset à ", nb_staturations / nb_current_request)
            CSP_link_load = np.zeros(numLinks)
            nb_staturations = 0
            nb_current_request = 0

        # input computation
        source = Traffic_Matrix[c][0]
        destination = Traffic_Matrix[c][1]
        bandwidth = Traffic_Matrix[c][2]
        for i in range(numLinks):
            x_data[c][i] = FLAGS.link_capacity - CSP_link_load[i]  # avaiable link load
        x_data[c][numLinks] = source
        x_data[c][numLinks + 1] = destination
        x_data[c][numLinks + 2] = bandwidth


        # computation of the shortest path
        GGG = G.copy()
        for i in range(numLinks):
            if (CSP_link_load[i] + bandwidth > FLAGS.link_capacity):
                GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = numLinks * 2
            else:
                GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = 1
        CSP_Path = YenKShortestPaths(GGG)
        SPtmp = CSP_Path.findFirstShortestPath(source, destination).nodeList


        # label computation
        y_tmp = np.zeros(FLAGS.k_path_generated + 1)
        y_tmp[0] = 1
        for i in range(FLAGS.k_path_generated ):
            if SPtmp == matrix_shortest_path[traffic_s_d_to_index(source,destination)][i] :
                y_tmp[0] = 0
                y_tmp[i + 1] = 1
                break
        # first definition of the label
        for i in range(FLAGS.k_path_generated + 1):
            y[c][i] = y_tmp[i]


        CSP_Fpath = (CSP_Path.findFirstShortestPath(source, destination)).nodeList

        #Check saturation
        CSP_saturation = False
        for i in range(len(CSP_Fpath) - 1):
            edge = (CSP_Fpath[i], CSP_Fpath[i + 1])
            if edge in GGG.edges():
                edge_index = GGG.edges().index(edge)
                if (bandwidth + CSP_link_load[edge_index] > GGG[CSP_Fpath[i]][CSP_Fpath[i + 1]]['capacity']):
                    CSP_saturation = True
                    break

        # when the network is saturated, we change the label
        if (CSP_saturation):
            nb_staturations += 1
            y[c][0] = 1
            for i in range(FLAGS.k_path_generated ):
                y[c][i+ 1] = 0
        else:
            CSP_avg_bd = CSP_avg_bd + bandwidth
            for i in range(len(CSP_Fpath) - 1):
                edge = (CSP_Fpath[i], CSP_Fpath[i + 1])
                if edge in GGG.edges():
                    edge_index = GGG.edges().index(edge)
                    CSP_link_load[edge_index] = CSP_link_load[edge_index] + bandwidth
    return x_data,  y

def k_shortest_path_multi_class(G, Traffic_Matrix):



    CSP_avg_bd = 0
    nb_staturations = 0
    nb_current_request = 0
    CSP_link_load = np.zeros(numLinks)

    C = len(Traffic_Matrix)


    y = np.zeros((C, 1))

    x_data = np.zeros((C, numLinks + 3))

    GGG = G.copy()
    for i in range(numLinks):
        GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = 1.0


    yksp = YenKShortestPaths(GGG)
    matrix_shortest_path = []

    # Generation of the matrix with the paths
    for i in range(FLAGS.network_node) :

        for j in range(FLAGS.network_node ):
            tmpLine = []
            if i != j :
                tmpLine.append(yksp.findFirstShortestPath(i,j).nodeList)
                # for each input we want FLAGS.k_path_generated labels
                for k in range(FLAGS.k_path_generated - 1 ):
                    tmpLine.append(yksp.getNextShortestPath().nodeList)
                matrix_shortest_path.append(tmpLine)

    for c in range(C):

        nb_current_request += 1

        # when we have more than FLAGS.percentage_of_congestion of congestion in our exemple
        # we reset the capacity of each link
        if ( nb_staturations / nb_current_request > FLAGS.percentage_of_congestion):
            CSP_link_load = np.zeros(numLinks)
            nb_staturations = 0
            nb_current_request = 0

        # input computation
        source = Traffic_Matrix[c][0]
        destination = Traffic_Matrix[c][1]
        bandwidth = Traffic_Matrix[c][2]
        for i in range(numLinks):
            x_data[c][i] = FLAGS.link_capacity - CSP_link_load[i]  # avaiable link load
        x_data[c][numLinks] = source
        x_data[c][numLinks + 1] = destination
        x_data[c][numLinks + 2] = bandwidth


        # computation of the shortest path
        GGG = G.copy()
        for i in range(numLinks):
            if (CSP_link_load[i] + bandwidth > FLAGS.link_capacity):
                GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = numLinks * 2
            else:
                GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = 1
        CSP_Path = YenKShortestPaths(GGG)
        SPtmp = CSP_Path.findFirstShortestPath(source, destination).nodeList


        # label computation
        for i in range( FLAGS.k_path_generated ):
            if SPtmp == matrix_shortest_path[traffic_s_d_to_index(source,destination)][i] :
                y[c][0] = i + 1
                break



        CSP_Fpath = (CSP_Path.findFirstShortestPath(source, destination)).nodeList

        #Check saturation
        CSP_saturation = False
        for i in range(len(CSP_Fpath) - 1):
            edge = (CSP_Fpath[i], CSP_Fpath[i + 1])
            if edge in GGG.edges():
                edge_index = GGG.edges().index(edge)
                if (bandwidth + CSP_link_load[edge_index] > GGG[CSP_Fpath[i]][CSP_Fpath[i + 1]]['capacity']):
                    CSP_saturation = True
                    break

        # when the network is saturated, we change the label
        if (CSP_saturation):
            nb_staturations += 1
            y[c][0] = 0
        else:
            CSP_avg_bd = CSP_avg_bd + bandwidth
            for i in range(len(CSP_Fpath) - 1):
                edge = (CSP_Fpath[i], CSP_Fpath[i + 1])
                if edge in GGG.edges():
                    edge_index = GGG.edges().index(edge)
                    CSP_link_load[edge_index] = CSP_link_load[edge_index] + bandwidth


    return x_data,  y



