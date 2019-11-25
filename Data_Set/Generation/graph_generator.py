#############################################
# graph generator for the GraphSAGE algorithm
# Pedro FOLETTO PIMENTA, lab I3S
# July 2018

#############################################
# usage:
# python graph_generator.py
# which sets feature dimension to 2 (default value)
# or
# python graph_generator.py [FEATURE_DIMENSION]

#############################################
# types of graphs : line, star, ring, grid, mesh
# num of nodes between 5 and 20

import os
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import time
import math
import tensorflow as tf
from Data_Set.Generation.YenKShortestPaths import YenKShortestPaths


flags = tf.app.flags
FLAGS = flags.FLAGS

def loadTopology(pathTopology, nbNode, nbLinks):
    file = open(pathTopology + "topology.json", "r")
    topology = json.load(file)
    return topology

def getKeysByValues(dictOfElements, listOfValues):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] in listOfValues:
            listOfKeys.append(item[0])
    return  listOfKeys

def get_k_hop_neighour(G, node, k):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    neighbours = []
    for i in range(k):
        for node, length in path_lengths.items():
            if (length == i + 2):
                if node not in neighbours:
                    neighbours.append(node)
    return neighbours

def get_one_hop_neighour(G, node):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    neighbours = []
    for node, length in path_lengths.items():
        if (length == 1):
            neighbours.append(node)
    return neighbours

def draw_graph(G, graph_pos, labels=None,
               node_size=60, node_color='blue', node_alpha=0.3,
               node_text_size=8,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size, 
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G,graph_pos,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
                            font_family=text_font)

#    if labels is None:
#        labels = range(len(G))
#
#    edge_labels = {tuple(graph[i]) : labels[i] for i in range(len(G)) }
#    print(edge_labels)
#
#    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels, 
#                                 label_pos=edge_text_pos)

    # show graph
    plt.show( )   


def genrateOneMeshGraph_nx(N, Degree,nodePositions_table, seed):
    G_undi = nx.gnm_random_graph(N, (Degree*N)/2, seed, False)
        #change to directed graph
    G = nx.DiGraph(G_undi)
    #plotting nx  
    ntwDict_keys=np.arange(N)
#    nodePositions_table = np.random.rand(N,2)
    #nodePositions_ntwDict = dict(zip(ntwDict_keys, nodePositions_table[:,(0,1)]))
    # if edge labels is not specified, numeric labels (0, 1, 2...) will be used
    #draw_graph(G, nodePositions_ntwDict)
    links = G.edges()
#    print(G.edges())
#    for edge in G.edges():
#        left_node = edge[0] 
#        right_node = edge[1]
#        print ('left_node',left_node,)
#        print ('right_node',right_node)
#        links.append([left_node,right_node])
#    print(links)
    return G, links
    
def generateALLMeshGraph(nbNodes, degree, nb_graph):
    ALL_graph_count = 0
    seed = 0
    ALL_G = []
    ALL_links = []
    nodePositions_table = np.random.rand(nbNodes,2)
    # 1) Generate the graphs
    while(ALL_graph_count < nb_graph):
        seed =  time.time() * 1000.0 
        print('seed', seed)
        #seed = random.seed(t)
#        seed = random.seed( ((t & 0xff000000) >> 24) +
#                    ((t & 0x00ff0000) >>  8) +
#                    ((t & 0x0000ff00) <<  8) +
#                    ((t & 0x000000ff) << 24)   )
        print('seed', seed)
        G, links = genrateOneMeshGraph_nx(nbNodes, degree, nodePositions_table, seed)
        G_in_ALL_G = False
        disc_G = False
        print('G_in_ALL_G', G_in_ALL_G)
        print('seed', seed)
        if len(nx.isolates(G)) > 0:
            disc_G = True
        if(ALL_graph_count > 0):
            for i in range(ALL_graph_count):
                if(nx.is_isomorphic(G, ALL_G[i])):
                    G_in_ALL_G = True
                    break
        print('G_in_ALL_G', G_in_ALL_G)
        if(not G_in_ALL_G and not disc_G):
            ALL_graph_count += 1
            print ('ALL_graph_count',ALL_graph_count)
            ALL_G.append(G)
            ALL_links.append(links)
    print("\n\n \n \n \n  ALL_graph_count " ,ALL_graph_count ,  "  nb_graph ", nb_graph)
    
    #2) - compute the adjacency and incindence matrices 
    #   - and path sets of the random  graphs
    #   - and save all in dictionary written into disk as json file
    for i in range(len(ALL_G)):
        nbLinks = int(nbNodes*degree)
        pathTopology = FLAGS.pathSourceData + str(nbNodes) + "node/" + str(degree) + "degree/" + str(i) + "_instance/"  
        # instansiation de la matrice d'adjacence
        adjMat = np.zeros((nbNodes,nbNodes), dtype=float)
        incMat = np.zeros((nbLinks,nbNodes), dtype=float)
        # calcul de la matrice d'adjacence
        edge_index = 0
        for edge in ALL_links[i]:
            adjMat[edge[0],edge[1]] = 1
            incMat[edge_index][edge[0]] = -1
            incMat[edge_index][edge[1]] = 1
            edge_index += 1
        list_paths, sd_vs_path_bin_table, link_vs_path_bin_table, node_vs_path_bin_table = computingPathSet(ALL_G[i])
        topology = {
            "links": ALL_links[i],
            "adjMat": adjMat.tolist(),
            "incMat": incMat.tolist(),
            "list_paths": list_paths, 
            "sd_vs_path_bin_table": sd_vs_path_bin_table.tolist(), 
            "link_vs_path_bin_table": link_vs_path_bin_table.tolist(),
            "node_vs_path_bin_table": node_vs_path_bin_table.tolist()
        }
        filename = pathTopology + "topology.json"
        if not os.path.exists(pathTopology):
            os.makedirs(pathTopology)
        with open(filename, 'w') as outfile:
            print("Done : " , filename)
            json.dump(topology, outfile)  
            outfile.close()
#        
#               graphPathisntance = graphPath + str(i) + "_instance/"
#        graphName = "links.txt"
#        with open(graphPathisntance + graphName, 'w') as outfile:
#            for j in range(len(graph_link_list)):
#                edge = graph_link_list[j]
#                #print ('edge',edge)
#                outfile.write(str(edge[0]) + ',' + str(edge[1]) + '\n')
#            outfile.close()                                                                     
#       with open( Folderpath + "list_paths.txt", 'wb') as f:
#        if not os.path.exists(Folderpath + "list_paths.txt"):
#            os.makedirs(Folderpath + "list_paths.txt")
#    #with open("Data_Set/" + FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/matriceShortestPath.txt", 'wb') as f:
#        pickle.dump(list_paths, f) 
        
    return  ALL_G, ALL_links

def computingPathSet(G):   
    GGG = G.copy()
    numLinks = GGG.number_of_edges()
    numNodes = len(GGG)
    for i in range(numLinks):
        GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['weight'] = 1.0
        GGG[GGG.edges()[i][0]][GGG.edges()[i][1]]['capacity'] = FLAGS.capacity
#    print('\n GGG.edges()', GGG.edges())
    
    yksp = YenKShortestPaths(GGG)
    list_paths = []
    sd_vs_path_bin_table = np.zeros((numNodes*numNodes, numNodes*numNodes*FLAGS.k_path_generated))
    link_vs_path_bin_table = np.zeros((numLinks, numNodes*numNodes*FLAGS.k_path_generated))
    node_vs_path_bin_table = np.zeros((numNodes, numNodes*numNodes*FLAGS.k_path_generated))
    
    # Generation of the matrix with the paths
    path = 0
    for i in range(numNodes) :
        for j in range(numNodes):
            tmpLine = []
            sd = i*numNodes + j
            #print('\n i , j ', i, j)
#            print('\n sd , path ', sd, path)
            if (i != j):
                if yksp.findFirstShortestPath(i,j) == None :
                    print('i',i,'j',j, ' are not connected')
                    continue
                sPath = yksp.findFirstShortestPath(i,j).nodeList
                tmpLine.append(sPath)
                sd_vs_path_bin_table[sd][path] = 1
                for n in range(len(sPath) - 1): #i source nodes in path
                    link_src = sPath[n]
                    link_dst = sPath[n+1]
                    node_vs_path_bin_table[link_src][path] = 1
                    node_vs_path_bin_table[link_dst][path] = 1
                    edge = (link_src, link_dst)
                    if edge in GGG.edges():
                        edge_index = GGG.edges().index(edge)
                        link_vs_path_bin_table[edge_index][path] = 1
                       # print('\n edge', edge)
                       # print('\n edge_index', edge_index)
                path += 1              
#                print('\n node list ', sPath)
#                print('\n node_vs_path_bin_table', node_vs_path_bin_table)
#                print('\n link_vs_path_bin_table', link_vs_path_bin_table)
#                print('\n sd_vs_path_bin_table', sd_vs_path_bin_table)

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
                        sd_vs_path_bin_table[sd][path] = 1
                        for n in range(len(kPATH.nodeList) - 1): #i source nodes in path
                            link_src = kPATH.nodeList[n]
                            link_dst = kPATH.nodeList[n+1]
                            node_vs_path_bin_table[link_src][path] = 1
                            node_vs_path_bin_table[link_dst][path] = 1
                            #print('\n (link_s,link_d)= ', link_src, link_dst)
                            edge = (link_src, link_dst)
                            if edge in GGG.edges():
                                edge_index = GGG.edges().index(edge)
                                link_vs_path_bin_table[edge_index][path] = 1
                                #print('\n edge_index ', edge_index)
                            #print('\n path ', path)
                        path += 1               
                        #print('\n node list ', kPATH.nodeList)
#                    print('\n node_vs_path_bin_table', node_vs_path_bin_table)
#                    print('\n link_vs_path_bin_table', link_vs_path_bin_table)
#                    print('\n sd_vs_path_bin_table', sd_vs_path_bin_table)
                
                    #print(k,yksp.getNextShortestPath())
                    #print(yksp.getNextShortestPath().nodeList)
                    #if(yksp.getNextShortestPath() == None):
                    #    print("is None")
                    #tmpLine.append(yksp.getNextShortestPath().nodeList)
                    #print('tmpLine',tmpLine)
                list_paths.append(tmpLine)
        
    return list_paths, sd_vs_path_bin_table, link_vs_path_bin_table, node_vs_path_bin_table  

def saveALLMeshGraph(links,N,degree):
    
    #generatedGraphsPath = "Data_Set/DataLinkToPathOH/"
    graphPath = FLAGS.pathSourceData + str(N) + "node_" + str(int(degree)) + "degree/"
    if not os.path.exists(graphPath):
        os.makedirs(graphPath)
    print('graphPath',graphPath)
    for i in range(len(links)):
        graph_link_list = links[i]
        #print ('graph_link_list',graph_link_list)
        graphPathisntance = graphPath + str(i) + "_instance/"
        if not os.path.exists(graphPathisntance):
            os.makedirs(graphPathisntance)
        graphName = "links.txt"
        with open(graphPathisntance + graphName, 'w') as outfile:
            for j in range(len(graph_link_list)):
                edge = graph_link_list[j]
                #print ('edge',edge)
                outfile.write(str(edge[0]) + ',' + str(edge[1]) + '\n')
            outfile.close()
            
        