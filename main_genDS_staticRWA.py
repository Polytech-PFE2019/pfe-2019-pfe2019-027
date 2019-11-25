#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:03:31 2019

@author: raparicio
"""
import sys
#sys.path.insert(0, "/Applications/CPLEX_Studio128/cplex/python/3.6/x86-64_osx")
#sys.path.insert(0, "/Users/raparicio/Applications/IBM/ILOG/CPLEX_Studio1262/cplex/python/2.7/x86-64_osx")
#sys.path.insert(0, "/Users/raparicio/Applications/IBM/ILOG/CPLEX_Studio1262/cplex/python/3.4/x86-64_osx")

import errno
import math

#import cplex
#from cplex.exceptions import CplexSolverError
#from cplex.exceptions import error_codes

import numpy as np
from scipy import sparse
np.set_printoptions(threshold=np.inf)
import tensorflow as tf
import os
import time
#from Data_Set.Generation.YenKShortestPaths import YenKShortestPaths
#from Data_Set.Generation.Dataset import generate_dataset_linkToWeight, generate_dataset_linkToPath
#from Data_Set.Generation.Topology import getTopologyWithPath
#from Data_Set.Generation.Traffic_generator import Traffic_Generator
from Data_Set.Generation.graph_generator import generateALLMeshGraph, loadTopology
from Data_Set.Generation.algorithms.RWA_SPF_FF import RWA_SPF_FF
from Data_Set.Generation.algorithms.ILP_oracle import ILP_oracle
#from Data_Set.Generation.algorithms.ILP_oracle_path import ILP_oracle_path

from Data_Set.Generation.Traffic_generator import lightpathMatrixGenerator,LPrequestTraceGenerator,load_dsLPMatrices,load_dynTraffTrace
from Data_Set.Generation.Dataset import generateDatasets

from sklearn.model_selection import train_test_split
import json
import pandas as pd
flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_boolean('linux', True,'if false, OSX')
flags.DEFINE_integer('nb_reqs', 5000, 'number of requests for a given avg traf matrix and a given topology')
flags.DEFINE_integer('nb_random_dynTraces', 32, 'total number of random dyn traces for a given traffic matrix.')
flags.DEFINE_integer('nb_random_matrices', 1, 'total number of random traffic matrices in data set for a given number of nodes.')
flags.DEFINE_integer('nb_random_graph', 1, 'nb_random_graph')
flags.DEFINE_integer('numWavelengths', 4, 'numWavelengthson each link.')
flags.DEFINE_integer('capacity', 1, 'maximum capacity on each link or WDM channel') #200
flags.DEFINE_integer('k_path_generated', 3, ' K shortes paths generated for the data set "LinkToPath" ')

flags.DEFINE_string('pathNewDataSet', "Data_Set/data/" , '')
flags.DEFINE_string('pathSourceData', "Data_Set/raw_data/", '')
flags.DEFINE_boolean('genRandomNetws', False, '')
flags.DEFINE_boolean('genRandomLPmatrices', False, '')
flags.DEFINE_boolean('genDynTraffTraces', True, '')
flags.DEFINE_boolean('genSolutions', True, 'Generate RWA soltions, i.e. Y labels for the raw DS')
flags.DEFINE_boolean('genDataset', True, 'Generate final datasets with the proper data representation')
flags.DEFINE_integer('state', 1, 'input data to the neural networks'
                                 '''
                                 0 is edge embedding state: L vectors + one N hot-vector,
                                     where each link vector = [w_0 ... w_W vol]_l, 
                                     where the N-hot vector is: vector(src) = - Vol; vector(dst) = + Vol
                                 1 is node embedding state: N vectors, each node vector = [lw_0 ...lw_LW vol]_n, 
                                     where vol > 0, if node = src; vol < 0, if node = dst, vol = 0, otherwise 
                                 
                               '''
                     )

def main():
    nodes_to_test = [10]
    #degrees_to_test = [3, 5, 8]
    degrees_to_test = [5]
    trafficType = 0 #Uniform traffic
    avgLPduration=1
    #maxUtilPerNodePair_to_test= [0.1, 0.3, 0.6, 0.9]
    maxUtilPerNodePair_to_test= [0.3]

    
    #1) Generate the nb_random_graph graph for a specific nb_node 
    if (FLAGS.genRandomNetws):
        for n in nodes_to_test :
            for degree  in degrees_to_test:
                generateALLMeshGraph(n, degree, FLAGS.nb_random_graph)
                #saveALLMeshGraph(ALL_links, n, degree)
    
    # - each sample is RWA for a lightpath matrix and a given topology
    
    #2) Generate "nb_random_matrices" LP matrices for each graph for a specific nb_node 
    if (FLAGS.genRandomLPmatrices):
        print(FLAGS.nb_random_matrices)
        #maxLPrequest = 10
        for n in nodes_to_test :
             pathname = FLAGS.pathSourceData + str(n) + "node/"   
             lightpathMatrixGenerator(FLAGS.nb_random_matrices, n, pathname, trafficType)
        print('LP_Traffic_Mat_Generator done')  
    
    #3) Generate for each LP matrix a dynamic traffic trace composed by a request list of LP establishments with the ligthpath durations
    if (FLAGS.genDynTraffTraces):
        for n in nodes_to_test :
            pathname = FLAGS.pathSourceData + str(n) + "node/"
            list_LPmatrices = load_dsLPMatrices(pathname, trafficType) # it returns a list of numpy nd arrays (0-1 LPs/unit time)
            for lpMat_i in range(FLAGS.nb_random_matrices):
                maxCapPerNodePair = FLAGS.k_path_generated*FLAGS.numWavelengths
                for maxUtilPerNodePair in maxUtilPerNodePair_to_test:
                    pathname = FLAGS.pathSourceData + str(n) + "node/" + str(lpMat_i) + "_traff_matrix/" + str(int(maxUtilPerNodePair*100)) + "_ut/" 
                    LPrequestTraceGenerator(FLAGS.nb_random_dynTraces, FLAGS.nb_reqs, pathname, list_LPmatrices[lpMat_i], avgLPduration, maxCapPerNodePair, maxUtilPerNodePair)
        print('LP_dyn_traff_generator done')  
        
    #4) Generate RWA Solutions (raw dataset)
    if (FLAGS.genSolutions):
        for n in nodes_to_test :
            for deg in degrees_to_test :
                numLinks = int(deg * n)
                for netw_i in range(FLAGS.nb_random_graph):
                    pathTopology = FLAGS.pathSourceData + str(n) + "node/" + str(deg) + "degree/" + str(netw_i) + "_instance/"   
                    topology = loadTopology(pathTopology, n, numLinks) # it returns a dict
                    # for each LP matrix, we find the complete RWA solution and we builf the raw data set
                    for lpMat_i in range(FLAGS.nb_random_matrices):
                        for maxUtilPerNodePair in maxUtilPerNodePair_to_test:
                            pathname = FLAGS.pathSourceData + str(n) + "node/" + str(lpMat_i) + "_traff_matrix/" + str(int(maxUtilPerNodePair*100)) + "_ut/" 
                            for i in range(FLAGS.nb_random_dynTraces):
                                arrivalsList = load_dynTraffTrace(pathname, i)
                                sol_pathname = pathTopology + str(lpMat_i) + "_traff_matrix/" + str(int(maxUtilPerNodePair*100)) + "_ut/trace_" + str(i) + "/"
                                
                                #RWA_SPF_FF(sol_pathname, topology, arrivalsList, FLAGS.numWavelengths)
                                ILP_oracle(sol_pathname, topology, arrivalsList, FLAGS.numWavelengths, "file")
        print('sols generated')  

    #5) Generate final dataset
    if (FLAGS.genDataset):
        for n in nodes_to_test :
            for deg in degrees_to_test :
                numLinks = int(deg * n)
                for netw_i in range(FLAGS.nb_random_graph):
                    topo_pathname = str(n) + "node/" + str(deg) + "degree/" + str(netw_i) + "_instance/"  
                    pathTopology = FLAGS.pathSourceData + topo_pathname
                    topology = loadTopology(pathTopology, n, numLinks) # it returns a dict
                    # for each LP matrix, we find the complete RWA solution and we builf the raw data set
                    for lpMat_i in range(FLAGS.nb_random_matrices):
                        for maxUtilPerNodePair in maxUtilPerNodePair_to_test:
                            traff_pathname = str(lpMat_i) + "_traff_matrix/" + str(int(maxUtilPerNodePair*100)) + "_ut/"
                            #algo_name = "RWA_SPF_FF"
                            algo_name = "ILP_oracle"
                            newDS_folder = FLAGS.pathNewDataSet
                            newDS_pathname =  topo_pathname + traff_pathname                      
                            generateDatasets(FLAGS.nb_random_dynTraces, traff_pathname, pathTopology, algo_name, newDS_folder, newDS_pathname, topology)
#                            for i in range(FLAGS.nb_random_dynTraces):
#                                traff_pathname = str(lpMat_i) + "_traff_matrix/" + str(int(maxUtilPerNodePair*100)) + "_ut/trace_" + str(i) + "/"
#
#                                generateDatasets(algo_sol, topology, newDS_folder, newDS_pathname, algo_name)
                        
main()  