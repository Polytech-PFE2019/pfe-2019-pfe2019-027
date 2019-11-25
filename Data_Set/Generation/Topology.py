"""
Read topology from 3-node, 5-node and 12-node networks folders.
Including: net-topology, get_outgoing_edges, get_incoming_edges, get_outgoing_nodes and get_incoming_nodes
"""
import networkx as nx
import tensorflow as tf
import numpy as np

#import sys
#import os
#print(sys.path)
#sys.path.append('/home/minju/Documents/Research/Code/Github/drl_csp/Data_Set/DataLinkToPathOH/11-node')
#sys.path.remove('/home/minju/Documents/Research/Code/Github/drl_csp/Data_Set/Generation')
#print(sys.path)


flags = tf.app.flags
FLAGS = flags.FLAGS
# return Topology from file
#'''+ FLAGS.dataset_name + "/"'''

def Topology():
    # TODO: add networks_name to a string in the folder_path
    file = open("Data_Set/DataLinkToPathOH/"  + str(FLAGS.network_node) + "-node/links.txt", "r")
    links = []
    for line in file.readlines():
        links.append([int(x) for x in line.split(',')])
    G = nx.DiGraph()
    for link in links:
        G.add_edge(link[0], link[1], weight=1, capacity=FLAGS.link_capacity)
    return G

def New_Topology():
    # TODO: add networks_name to a string in the folder_path
    file = open("Data_Set/" + FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node-valid/links.txt", "w+")
    #file = open("../../Data_Set/" + FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/links.txt", "r")
    #file = open("/home/minju/Documents/Research/Code/Github/drl_csp/Data_Set/DataLinkToPathOH/11-node/links.txt","r")
    links = []
    for line in file.readlines():
        links.append([int(x) for x in line.split(',')])
    G = nx.DiGraph()
    for link in links:
        G.add_edge(link[0], link[1], weight=1, capacity=FLAGS.link_capacity)
    return G

def Different_Topology(Folderpath):
    # TODO: add networks_name to a string in the folder_path
    #Folderpath = "Data_Set/" + FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "node-" + str(degree) + "degree/"
    #filename = str(n) + "_links.txt"
    filename = "links.txt"
    file = open(Folderpath+ filename, "r")
    #file = open("../../Data_Set/" + FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/links.txt", "r")
    #file = open("/home/minju/Documents/Research/Code/Github/drl_csp/Data_Set/DataLinkToPathOH/11-node/links.txt","r")
    links = []
    for line in file.readlines():
        links.append([int(x) for x in line.split(',')])
    G = nx.DiGraph()
    for link in links:
        G.add_edge(link[0], link[1], weight=1, capacity=FLAGS.link_capacity)
    return G

def getTopologyWithPath(Folderpath):

    file = open(Folderpath, "r")
    links = []
    for line in file.readlines():
        links.append([int(x) for x in line.split(',')])
    G = nx.DiGraph()
    for link in links:
        G.add_edge(link[0], link[1], weight=1, capacity=FLAGS.link_capacity)
    return G


def get_outgoing_edges(G, node_name):
    outgoing_edges = []
    for edge in G.edges():
        if edge[0] == node_name:
            outgoing_edges.append(edge)
    return outgoing_edges

def get_incoming_edges(G,node_name):
    incoming_edges = []
    for edge in G.edges():
        if edge[1] == node_name:
            incoming_edges.append(edge)
    return incoming_edges


def get_neibhour_links_matrix_mean(G):
    neighour_links_matrix = np.zeros((FLAGS.network_link, FLAGS.network_link), dtype=np.float32)

    for link in G.edges():
        incoming_links = get_incoming_links_link(G, link)
        link_index = get_link_index(G,link)
        for i in range(len(incoming_links)):
            neighour_links_matrix[get_link_index(G,incoming_links[i])][link_index] = 1
        outgoing_links = get_outgoing_links_link(G, link)
        for j in range(len(outgoing_links)):
            neighour_links_matrix[get_link_index(G,outgoing_links[j])][link_index] = 1

    # print('before',neighour_links_matrix)
    # with reduce_mean
    sum_column = np.zeros(FLAGS.network_link)
    for i in range(FLAGS.network_link):
        for j in range(FLAGS.network_link):
            sum_column[i] += neighour_links_matrix[j][i]

    for i in range(FLAGS.network_link):
        for j in range(FLAGS.network_link):
            if (sum_column[i] > 0):
                neighour_links_matrix[j][i] = 1.0 * neighour_links_matrix[j][i] / sum_column[i]
    # print('after',neighour_links_matrix)
    return neighour_links_matrix

def get_neibhour_self_links_matrix_gcn(G):
    neighour_self_links_matrix = np.zeros((FLAGS.network_link, FLAGS.network_link), dtype=np.float32)
    for link in G.edges():
        link_index = get_link_index(G, link)
        incoming_links = get_incoming_links_link(G, link)
        neighour_self_links_matrix[link_index][link_index] = 1
        for i in range(len(incoming_links)):
            neighour_self_links_matrix[get_link_index(G, incoming_links[i])][link_index] = 1
        outgoing_links = get_outgoing_links_link(G, link)
        for j in range(len(outgoing_links)):
            neighour_self_links_matrix[get_link_index(G, outgoing_links[j])][link_index] = 1

    # print('before',neighour_links_matrix)
    # with reduce_mean
    sum_column = np.zeros(FLAGS.network_link)
    for i in range(FLAGS.network_link):
        for j in range(FLAGS.network_link):
            sum_column[i] += neighour_self_links_matrix[j][i]

    for i in range(FLAGS.network_link):
        for j in range(FLAGS.network_link):
            if (sum_column[i] > 0):
                neighour_self_links_matrix[j][i] = 1.0 * neighour_self_links_matrix[j][i] / sum_column[i]

    # print('after',neighour_nodes_matrix)
    return neighour_self_links_matrix

def get_neibhour_self_links_matrix_rcn(G):
    neighour_self_links_matrix_in = np.zeros((FLAGS.network_link, FLAGS.network_link), dtype=np.float32)
    neighour_self_linkes_matrix_out = np.zeros((FLAGS.network_link, FLAGS.network_link), dtype=np.float32)
    for link in G.edges():
        link_index = get_link_index(G, link)
        incoming_links = get_incoming_links_link(G, link)
        for i in range(len(incoming_links)):
            neighour_self_links_matrix_in[get_link_index(G, incoming_links[i])][link_index] = 1
        outgoing_links = get_outgoing_links_link(G, link)
        for j in range(len(outgoing_links)):
            neighour_self_linkes_matrix_out[get_link_index(G, outgoing_links[j])][link_index] = 1

    sum_column = np.zeros(FLAGS.network_link)
    for i in range(FLAGS.network_link):
        for j in range(FLAGS.network_link):
            sum_column[i] += neighour_self_links_matrix_in[j][i]

    for i in range(FLAGS.network_link):
        for j in range(FLAGS.network_link):
            if (sum_column[i] > 0):
                neighour_self_links_matrix_in[j][i] = 1.0 * neighour_self_links_matrix_in[j][i] / sum_column[i]

    sum_column_out = np.zeros(FLAGS.network_link)
    for i in range(FLAGS.network_link):
        for j in range(FLAGS.network_link):
            sum_column_out[i] += neighour_self_linkes_matrix_out[j][i]

    for i in range(FLAGS.network_link):
        for j in range(FLAGS.network_link):
            if (sum_column[i] > 0):
                neighour_self_linkes_matrix_out[j][i] = 1.0 * neighour_self_linkes_matrix_out[j][i] / sum_column_out[i]

    return neighour_self_links_matrix_in, neighour_self_linkes_matrix_out


def get_outgoing_nodes(G,node_name):
    outgoing_nodes = []
    for edge in G.edges():
        if edge[0] == node_name:
            outgoing_nodes.append(edge[1])
    return outgoing_nodes

def get_incoming_nodes(G,node_name):
    incoming_nodes = []
    for edge in G.edges():
        if edge[1] == node_name:
            incoming_nodes.append(edge[0])
    return incoming_nodes


def get_neibhour_nodes_matrix_mean(G):
    neighour_nodes_matrix = np.zeros((FLAGS.network_node, FLAGS.network_node), dtype=np.float32)
    for node in G.nodes():
        incoming_nodes = get_incoming_nodes(G, node)
        for i in range(len(incoming_nodes)):
            neighour_nodes_matrix[incoming_nodes[i]][node] = 1
        outgoing_nodes = get_outgoing_nodes(G, node)
        for j in range(len(outgoing_nodes)):
            neighour_nodes_matrix[outgoing_nodes[j]][node] = 1

    # print('before',neighour_nodes_matrix)
    # with reduce_mean
    sum_column = np.zeros(FLAGS.network_node)
    for i in range(FLAGS.network_node):
        for j in range(FLAGS.network_node):
            sum_column[i] += neighour_nodes_matrix[j][i]

    for i in range(FLAGS.network_node):
        for j in range(FLAGS.network_node):
            if (sum_column[i] > 0):
                neighour_nodes_matrix[j][i] = 1.0 * neighour_nodes_matrix[j][i] / sum_column[i]

    # print('after',neighour_nodes_matrix)
    return neighour_nodes_matrix


def get_neibhour_self_nodes_matrix_gcn(G):
    neighour_self_nodes_matrix = np.zeros((FLAGS.network_node, FLAGS.network_node), dtype=np.float32)
    for node in G.nodes():
        incoming_nodes = get_incoming_nodes(G, node)
        neighour_self_nodes_matrix[node][node] = 1
        for i in range(len(incoming_nodes)):
            neighour_self_nodes_matrix[incoming_nodes[i]][node] = 1
        outgoing_nodes = get_outgoing_nodes(G, node)
        for j in range(len(outgoing_nodes)):
            neighour_self_nodes_matrix[outgoing_nodes[j]][node] = 1

    # print('before',neighour_nodes_matrix)
    # with reduce_mean
    sum_column = np.zeros(FLAGS.network_node)
    for i in range(FLAGS.network_node):
        for j in range(FLAGS.network_node):
            sum_column[i] += neighour_self_nodes_matrix[j][i]

    for i in range(FLAGS.network_node):
        for j in range(FLAGS.network_node):
            if (sum_column[i] > 0):
                neighour_self_nodes_matrix[j][i] = 1.0 * neighour_self_nodes_matrix[j][i] / sum_column[i]

    # print('after',neighour_nodes_matrix)
    return neighour_self_nodes_matrix


def get_neibhour_self_nodes_matrix_rcn(G):
    neighour_self_nodes_matrix_in = np.zeros((FLAGS.network_node, FLAGS.network_node), dtype=np.float32)
    neighour_self_nodes_matrix_out = np.zeros((FLAGS.network_node, FLAGS.network_node), dtype=np.float32)

    for node in G.nodes():
        incoming_nodes = get_incoming_nodes(G, node)
        for i in range(len(incoming_nodes)):
            neighour_self_nodes_matrix_in[incoming_nodes[i]][node] = 1

        outgoing_nodes = get_outgoing_nodes(G, node)
        for j in range(len(outgoing_nodes)):
            neighour_self_nodes_matrix_out[outgoing_nodes[j]][node] = 1

    # print('before',neighour_nodes_matrix)
    # with reduce_mean
    sum_column = np.zeros(FLAGS.network_node)
    for i in range(FLAGS.network_node):
        for j in range(FLAGS.network_node):
            sum_column[i] += neighour_self_nodes_matrix_in[j][i]

    for i in range(FLAGS.network_node):
        for j in range(FLAGS.network_node):
            if (sum_column[i] > 0):
                neighour_self_nodes_matrix_in[j][i] = 1.0 * neighour_self_nodes_matrix_in[j][i] / sum_column[i]

    sum_column_out = np.zeros(FLAGS.network_node)
    for i in range(FLAGS.network_node):
        for j in range(FLAGS.network_node):
            sum_column_out[i] += neighour_self_nodes_matrix_out[j][i]

    for i in range(FLAGS.network_node):
        for j in range(FLAGS.network_node):
            if (sum_column[i] > 0):
                neighour_self_nodes_matrix_out[j][i] = 1.0 * neighour_self_nodes_matrix_out[j][i] / sum_column_out[i]

    # print('after',neighour_nodes_matrix)
    return neighour_self_nodes_matrix_in, neighour_self_nodes_matrix_out

def get_link_index(G, link):
    link_index = -1
    for i in range(FLAGS.network_link):
        if (link == G.edges()[i]) :
            link_index = i
            break

    return link_index

def link_neighbour(link):
    """
    :param link: the link information in G.
    :return: all the neighbour links.
    """
    neighbour_link_set = []
    G = Topology()
    left_node = link[0]
    right_node = link[1]
    set_incoming = get_incoming_edges(G, left_node)
    for edge in set_incoming:
        # if(edge[0]!=right_node and edge[1]!=left_node):
        neighbour_link_set.append(edge)
    neighbour_link_set.remove((right_node, left_node))
    set_outgoing = get_outgoing_edges(G, right_node)
    for edge in set_outgoing:
        # if(edge[0]!=right_node and edge[1]!=left_node):
        neighbour_link_set.append(edge)
    neighbour_link_set.remove((right_node, left_node))

    return neighbour_link_set

def get_outgoing_links_link(G, link):
    left_node = link[0]
    right_node = link[1]
    outgoing_edges = get_outgoing_edges(G, right_node)
    outgoing_edges.remove((right_node, left_node))
    return outgoing_edges

def get_incoming_links_link(G, link):
    left_node = link[0]
    right_node = link[1]
    # print('left_node',left_node,'right_node',right_node)
    incoming_edges = get_incoming_edges(G, left_node)
    # print(incoming_edges)
    incoming_edges.remove((right_node, left_node))
    return incoming_edges


def traffic_to_index():
    link_index_relationship = []
    for s in range(FLAGS.network_node):
        for d in range(FLAGS.network_node):
            if(s!=d):
                link_index_relationship.append([s,d])
    return link_index_relationship

def traffic_s_d_to_index(s,d):
    link_index_relationship = traffic_to_index()
    #print(link_index_relationship)
    index = -1
    for i in range(len(link_index_relationship)):
        if link_index_relationship[i][0] == s and link_index_relationship[i][1] == d:
            index = i
            break
    return index

#def adj2inc (adjacencyMatrix):
#    
#    
#    
#    return incidenceMatrix

