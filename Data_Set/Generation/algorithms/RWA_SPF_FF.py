#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:51:21 2019

@author: raparicio
"""
import os
#import time
import numpy as np
np.set_printoptions(threshold=np.inf)
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
import json

def RWA_SPF_FF(sol_pathname,topology, arrivalsList, numWavelengths):
    #from lists to nd array again
    arrivalsList = np.array(arrivalsList) # 4 cols: s, d, arriv_time, duration
    sd_vs_path_bin_table = np.array(topology['sd_vs_path_bin_table'])
    link_vs_path_bin_table = np.array(topology['link_vs_path_bin_table'])
    node_vs_path_bin_table = np.array(topology['node_vs_path_bin_table'])
    
    numNodes = node_vs_path_bin_table.shape[0]
    numLinks = link_vs_path_bin_table.shape[0]
 #   numPaths = link_vs_path_bin_table.shape[1]
    numPaths_per_nodepair = FLAGS.k_path_generated
  #  numNodePairs = numNodes*numNodes
    numReqs = arrivalsList.shape[0]
    
    y_data = np.zeros((numReqs, 1+ numPaths_per_nodepair*numWavelengths)) # first position [0] is no_solution
    y_linkBased_data = np.zeros((numReqs, numLinks*numWavelengths)) # 1 if req used lw, o otherwise   
    x_data = np.zeros((numReqs, numLinks*numWavelengths + 3)) # 1 if lw is free  
    link_wavelength_state = np.zeros((numLinks, numWavelengths)) # 1 if (l,w) is used

    numAcceptLPs = 0
    numBlockedLPs = 0
    departuresList = np.zeros((0,2)) # 2 cols: reqID, duration
    
    for req in range(numReqs) : 
        acceptedReq = False
        s = int(arrivalsList[req,0])
        d = int(arrivalsList[req,1])
        sd = s*numNodes + d
        arrivTime = arrivalsList[req,2]
        duration = arrivalsList[req,3]
        
        # Purging departures taking place between two consecutive arrivals 
        #asc_row_ind = np.argsort(departuresList[:,1], axis=None) 
        #departuresList = departuresList[asc_row_ind,:]
        departingReqs_ind = np.argwhere(departuresList[:,1] < arrivTime)
        numDepartingReqs = departingReqs_ind.size
        if numDepartingReqs > 0:
            requestsToRelease = departuresList[departingReqs_ind ,0]
            for reqToRelease_idx in range(numDepartingReqs):
                reqToRelease = int(requestsToRelease[reqToRelease_idx])
                linkWavelengthsToRelease = np.nonzero(y_linkBased_data[reqToRelease,:])[0]
                linkToRel,wavelToRel = np.divmod(linkWavelengthsToRelease, numWavelengths)
                link_wavelength_state[linkToRel, wavelToRel[0]] = 0
            departuresList = np.delete(departuresList, departingReqs_ind, axis=0)
                
        candPathIds = np.nonzero(sd_vs_path_bin_table[sd])[0]
        numCandPaths_per_nodepair = candPathIds.size
        if (numCandPaths_per_nodepair==0): continue
        for local_pathId in range(numCandPaths_per_nodepair): #0....K
            pathId = candPathIds[local_pathId]
            pathLinkIds = np.nonzero(link_vs_path_bin_table[:,pathId])[0]
            for wavelengthId in range(numWavelengths):
                if np.sum(link_wavelength_state[pathLinkIds,wavelengthId]) == 0 :
                    # the RWA is feasible
                    y_data[req, 1+ local_pathId*numWavelengths + wavelengthId] = 1
                    y_linkBased_data[req, pathLinkIds*numWavelengths + wavelengthId] = 1
                    # we capture the current state of available resources before LP establishment
                    x_data[req, pathLinkIds*numWavelengths + wavelengthId] = 1 - link_wavelength_state[pathLinkIds, wavelengthId] # free capacity
                    link_wavelength_state[pathLinkIds, wavelengthId] = 1 # the resources are now used 
                    # we capture info about current arrival
                    x_data[req, numLinks*numWavelengths + 0] = s
                    x_data[req, numLinks*numWavelengths + 1] = d
                    x_data[req, numLinks*numWavelengths + 2] = 1
                    # we capture the departure time of this request   
                    departuresList = np.append(departuresList, [[req, arrivTime+duration]], axis=0)

                    numAcceptLPs += 1
                    acceptedReq = True
                    break

            if acceptedReq : break
        
        if acceptedReq == False :  # blocked Req
            y_data[req, 0] = 1
            x_data[req, numLinks*numWavelengths + 0] = s
            x_data[req, numLinks*numWavelengths + 1] = d
            x_data[req, numLinks*numWavelengths + 2] = 1
            numBlockedLPs += 1

    print('\n totalNumLPs: ', numReqs)
    for local_pathId in range(numPaths_per_nodepair) :
        print('estNumLPs of path ', local_pathId, ': ', np.sum(y_data[:,1 + local_pathId*numWavelengths:1+(local_pathId+1)*numWavelengths ]) )    
    print('blockedNumLPs: ', np.sum(y_data[:,0]))

    dataset = {}
    dataset['X'] = x_data.tolist()
    dataset['y'] = y_data.tolist()
    dataset['topology'] = y_data.tolist()

    full_filename = sol_pathname + "RWA_SPF_FF.json"
    if not os.path.exists(sol_pathname):
        os.makedirs(sol_pathname)
    with open(full_filename, 'w') as outfile:
        print("Done : " , full_filename)
        json.dump(dataset, outfile)  
        outfile.close()