"""
Generate and save the traffic data for the simulation:

synthetic traffic and uniform traffic
"""
import os
import time
import json
import numpy as np
import random
import math

from Data_Set.Generation.Topology import Different_Topology

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


# return traffic instances from file
def load_dsLPMatrices(pathname, trafficType):
    if trafficType == 0:
        filename = "LPmatrices_uniform.json"
    elif trafficType == 1:
        filename = "LPmatrices_zipflike.json"

    file = open(pathname + filename, "r")
    list_LPmatrices = json.load(file)
    return list_LPmatrices

def load_dynTraffTrace(pathname, i):
    full_filename = pathname + "arrival_list_" + str(i) +".json"
    file = open(full_filename, "r")
    arrivalsList = json.load(file)
    return arrivalsList

def LPrequestTraceGenerator(nb_random_dynTraces, numReqs, pathname, LPmatrix, avgLPduration, maxCapPerNodePair, maxUtilPerNodePair):
    # we assume a Poisson Proces:
    #   - the duration of a lightpath follows an exponential ditribution of average "avgLPduration" 
    #   - the Inter-Arrival Time (IAT) of LPs from s to d also follows an exponential distribution 
    #       of average "1/LPmatrix(s,d)" 
    # the time units are arbitrary
    # basic relationship with network load :
    #  lambda_sd/mu = rho_sd = avg nb reqs at anytime btw s and d <= P_sd * W = maxCapPerNodePair
    #  the max capacity btw a node par (s-->d) is P_sd * W
    #  util_sd = rho_sd / (P_sd * W) = (avg nb reqs at anytime btw s and d) / maxCapPerNodePair
    #  lambda_sd ~ U[0,1) at the input. to obtain the demanded "utilisation," we will scale lambda_sd
    #  rho_sd = util_sd * (P_sd * W) = lambda_sd/mu
    #  lambda_sd = mu * util_sd * (P_sd * W) 
    #  sd_max = argmax_{sd}(lambda_sd) 
    #  new_lambda_sd_max = mu * maxUtil * (P_sd * W)
    #  scale = new_lambda_sd_max/lambda_sd_max = (mu * maxUtil * (P_sd * W))/mu * util_sd_max * (P_sd * W) = maxUtil/util_sd_max
    #  scale = maxUtil/(rho_sd_max / (P_sd_max * W)) =  maxUtil/((lambda_sd_max/mu ) / (P_sd_max * W)) 
    #  scale = (maxUtil * P_sd_max * W ) / (lambda_sd_max*avgLPduration) 

    LPmatrix = np.array(LPmatrix)    
    scale = np.divide(maxUtilPerNodePair*maxCapPerNodePair,LPmatrix.max()*avgLPduration)
    LPmatrix = np.multiply(scale,LPmatrix)  
    #print('LPmatrix', LPmatrix)
    totalOfferedTraffic = LPmatrix.sum() # Reqs /time unit
    totalSimTime = numReqs/totalOfferedTraffic
    
    actual_numReqs = np.floor(np.multiply(totalSimTime,LPmatrix)).sum()
    augmentedSimTime = np.ceil(numReqs/actual_numReqs)*totalSimTime
    
    #print('actual_numReqs', actual_numReqs)
    #print('totalSimTime ', totalSimTime)
    #print('augmentedSimTime ', augmentedSimTime)
    
    for i in range(nb_random_dynTraces):
        np.random.seed(int(time.time()))
        numNodes = LPmatrix.shape[0]
    
        arrivalsList = np.zeros((0,4)) # 4 cols: s, d, arriv_time, duration
        for s in range(numNodes):
            for d in range(numNodes): 
                if s==d: continue
                numReqs_thisNodePair = int(np.floor(augmentedSimTime*LPmatrix[s,d]))
                if numReqs_thisNodePair == 0: continue
                arrivalsList_thisNodePair = np.zeros((numReqs_thisNodePair,4)) # 4 cols: s, d, arriv_time, duration
                inter_arriv_times_vct = np.random.exponential(1/LPmatrix[s,d], size = (numReqs_thisNodePair))
                arrivalsList_thisNodePair[:,0] = s
                arrivalsList_thisNodePair[:,1] = d
                arrivalsList_thisNodePair[:,2] = np.cumsum(inter_arriv_times_vct)
                #print('arrivalsList_thisNodePair= ', arrivalsList_thisNodePair)
                arrivalsList = np.concatenate((arrivalsList, arrivalsList_thisNodePair), axis=0)
                
        asc_row_ind = np.argsort(arrivalsList[:,2], axis=None) 
        arrivalsList = arrivalsList[asc_row_ind[0:numReqs],:]
        reqDuration = np.random.exponential(avgLPduration, size = (numReqs))
        arrivalsList[:,3] = reqDuration    
        print('arrivalsList= ', arrivalsList)
        print('actual_numReqs', actual_numReqs)
        print('totalSimTime ', totalSimTime)
        print('augmentedSimTime ', augmentedSimTime)
        full_filename = pathname + "arrival_list_" + str(i) +".json"
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        with open(full_filename, 'w') as outfile:
            print("Done : " , full_filename)
            json.dump(arrivalsList.tolist(), outfile)  
            outfile.close()
    
def lightpathMatrixGenerator(numMatrices, numNodes, pathname, trafficType):
    # Generates a lightmatrix where num LP(s,d) follows a uniform distr [0,1)
    list_LPmatrices = []
    for i in range(numMatrices):
        np.random.seed(int(time.time()))
        if trafficType == 0:
            brutLPmatrix = np.random.rand(numNodes, numNodes)
            #brutLPmatrix = np.random.randint(maxLPrequest+1, size=(numNodes, numNodes))
            filename = "LPmatrices_uniform.json"
        elif trafficType == 1:
            #val = (maxLPrequest/64)*np.array([1, 2, 8, 32, 64])
            val = (1/64)*np.array([1, 2, 8, 32, 64])
            probas = np.array([450, 350, 125, 50, 25])/1000
            brutLPmatrix = np.random.choice(val, size=(numNodes, numNodes), p = probas)    
            filename = "LPmatrices_zipflike.json"    
        LPmatrix = np.multiply(brutLPmatrix, np.logical_not(np.eye(numNodes, dtype=int)))
        list_LPmatrices.append(LPmatrix.tolist())
        print('\n LPmatrix \n', LPmatrix)
        
    full_filename = pathname + filename
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    with open(full_filename, 'w') as outfile:
        print("Done : " , full_filename)
        json.dump(list_LPmatrices, outfile)  
        outfile.close()
    
#    with open(pathname + filename, 'w') as outfile:
#        print("Done : " , pathname  + filename)
#        json.dump(list_LPmatrices, outfile)



def Traffic_Generator(Folderpath, Random_seed, C):
    G = Different_Topology(Folderpath)
    # ==============================
    # generate synthetic traffic_scenario 1:
    # ==============================
    #file1 = open("Data_Set/" +FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/" +"synthetictraffic.txt", "w")
    file1 = open(Folderpath + "synthetictraffic.txt", "w+")
    comod_matrix_1 = np.zeros((C, 3))  # no. commodity, bandwidth type, types of service bandwidth is 5
    random.seed(Random_seed)
    Total_NODE = len(G.nodes())
    #print("Total_NODE ", Total_NODE)
    Node_list = [i for i in range(Total_NODE)]
    Source_list_length = int(math.ceil(0.2 * Total_NODE))
    #print('Node_list',Node_list,'Source_list_length',Source_list_length)
    Source_list = random.sample(Node_list, Source_list_length)
    #print('Node_list',Node_list)
    for i in range(Source_list_length):
        Node_list.remove(Source_list[i])

    #print('Source_list',Source_list,'Dest_list',Node_list)

    weighted1_choices = [(1, 450), (2, 350), (8, 125), (32, 50), (64, 25)]
    #weighted1_choices = [(0, 450), (1, 350), (2, 125), (3, 50), (4, 25)]
    population1 = [val for val, cnt in weighted1_choices for i in range(cnt)]

    for c in range(C):
        if(c < C * 0.8):
            #print("Source_list " , Source_list)
            s = random.sample(Source_list, 1)[0]
        else:
            s = random.sample(Node_list, 1)[0]
        Node_list_for_s = [i for i in range(Total_NODE)]
        Node_list_for_s.remove(s)
        comod_matrix_1[c][0] = int(s)
        comod_matrix_1[c][1] = int(random.sample(Node_list_for_s, 1)[0])
        comod_matrix_1[c][2] = random.choice(population1)


    column1, column2, column3 = zip(*comod_matrix_1)
    column1 = list(column1)
    column2 = list(column2)
    column3 = list(column3)
    random.seed(Random_seed+1)
    random.shuffle(column1)
    random.seed(Random_seed+1)
    random.shuffle(column2)
    random.seed(Random_seed+1)
    random.shuffle(column3)

    for dex in range(C):
        file1.write("%d %d %d" % (column1[dex], column2[dex], column3[dex]))
        file1.write(str('\n'))
    file1.close()

    # ==============================
    # generate uniform traffic_scenario 2:
    # ==============================

    #file2 = open("Data_Set/" +FLAGS.dataset_name + "/" + str(FLAGS.network_node) + "-node/" + "uniformtraffic.txt", "w")
    file2 = open(Folderpath + "uniformtraffic.txt", "w")

    comod_matrix_2 = np.zeros((C, 3))  # no. commodity, bandwidth type, types of service bandwidth is 5
    random.seed(Random_seed+2)
    weighted2_choices = [(1, 1), (1, 1)]
    #weighted2_choices = [(1, 1), (2, 1), (8, 1), (32, 1), (64, 1)]
    population2 = [val for val, cnt in weighted2_choices for i in range(cnt)]
    for cc in range(C):
        Node_list2 = [i for i in range(Total_NODE)]
        ss = random.sample(Node_list2, 1)[0]
        Node_list2.remove(ss)
        comod_matrix_2[cc][0] = int(ss)
        comod_matrix_2[cc][1] = int(random.sample(Node_list2, 1)[0])
        comod_matrix_2[cc][2] = random.choice(population2)

    column11, column22, column33 = zip(*comod_matrix_2)
    column11 = list(column11)
    column22 = list(column22)
    column33 = list(column33)
    random.seed(Random_seed+3)
    random.shuffle(column11)
    random.seed(Random_seed+3)
    random.shuffle(column22)
    random.seed(Random_seed+3)
    random.shuffle(column33)

    for dex in range(C):
        file2.write("%d %d %d" % (column11[dex], column22[dex], column33[dex]))
        file2.write(str('\n'))
    file1.close()


if __name__ == '__main__':
    tf.app.run()
