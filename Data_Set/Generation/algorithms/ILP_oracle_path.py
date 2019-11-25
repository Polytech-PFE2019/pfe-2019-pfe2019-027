#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:05:49 2019

@author: raparicio
"""

import sys
import gc
import itertools as it
#Linux
cpx_path = "/opt/ibm/ILOG"
cpx_os_path = "/x86-64_linux"
cpx_python_path = "/CPLEX_Studio128/cplex/python/3.6"
sys.path.insert(0, cpx_path + cpx_python_path + cpx_os_path)
 #mac
cpx_path = "/Applications"
cpx_os_path = "/x86-64_osx"
sys.path.insert(0, cpx_path + cpx_python_path + cpx_os_path)

import cplex
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
import os
#import time
import numpy as np
from scipy import sparse
np.set_printoptions(threshold=np.inf)
import json
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


def ILP_oracle_path(sol_pathname,topology, arrivalsList, numWavelengths, strfile):

    #from lists to nd array again
    arrivalsList = np.array(arrivalsList) # 4 cols: s, d, arriv_time, duration
    sd_vs_path_bin_table = np.array(topology['sd_vs_path_bin_table'])
    link_vs_path_bin_table = np.array(topology['link_vs_path_bin_table'])
    node_vs_path_bin_table = np.array(topology['node_vs_path_bin_table'])
    
    N = node_vs_path_bin_table.shape[0] #numNodes
    L = link_vs_path_bin_table.shape[0] #numLinks
    P = link_vs_path_bin_table.shape[1] #numPaths
    numPaths_per_nodepair = FLAGS.k_path_generated
  #  numNodePairs = numNodes*numNodes
    R = arrivalsList.shape[0] #numReqs
    W = numWavelengths
    
    reqSrcs = arrivalsList[:,0].flatten().astype(int)# requests sources
    reqDsts = arrivalsList[:,1].flatten().astype(int) # requests destinations
    reqArrivTimes = arrivalsList[:,2] # requests arrival ties
    reqDurations = arrivalsList[:,3] # requests durations
    reqDepartTimes = reqArrivTimes+reqDurations
    lastDeparture = np.amax(reqDepartTimes)
    interArrivTimes = np.append(reqArrivTimes[1:],lastDeparture) - reqArrivTimes
    reqSDs = N*reqSrcs + reqDsts
    req_vs_path_bin_table = sparse.csr_matrix(sd_vs_path_bin_table[reqSDs,:])
    link_vs_path_bin_table = sparse.csr_matrix(link_vs_path_bin_table)
            
    y_data = np.zeros((R, 1+ numPaths_per_nodepair*W)) # first position [0] is no_solution
    x_data = np.zeros((R, L*W + 3)) # 1 if lw is free  
     
    # Create a new (empty) model and populate it below.
    model = cplex.Cplex()
    model.objective.set_sense(model.objective.sense.maximize)
    model.parameters.mip.tolerances.mipgap.set(float(0.05))
    model.parameters.write_file("test.prm")

    print("Beginning of variables \n")
    # Add the x_rpw variables
    num_x_rpw = R*P*W
    #obj = np.kron(reqDurations, np.ones((P*W))).tolist()
    obj = [1.0] * num_x_rpw
    lb = [0.0] * num_x_rpw
    ub_x_rpw = sparse.kron(req_vs_path_bin_table.reshape((R*P,1)), np.ones((W,1)) ) 
    ub = ub_x_rpw.toarray().flatten().tolist()
    #ub = [1.0] * num_y_rpw
    types = ["B"]*num_x_rpw 
    # set up names of variables
    names = ["x_%s_%s_%s" % (r,p,w) for r in range(R) for p in range(P) for w in range(W)] 
    # add variables to the model and store indices as a list        
    #indices_x_rpw = list(model.variables.add(obj, lb, ub, types, names))
    model.variables.add(obj, lb, ub, types, names)
    del obj, lb, ub, types, names
#    gc.collect()
    del topology, arrivalsList, req_vs_path_bin_table
#    gc.collect()
    
    # Compute set of colliding requests
    req_vs_collid_bin_table = sparse.lil_matrix((R,R))
    for r in range(R):
        livingReqs = np.nonzero((reqArrivTimes <= reqArrivTimes[r]) & (reqArrivTimes[r] <= reqDepartTimes))[0]
        req_vs_collid_bin_table[r,livingReqs] = 1
    req_vs_collid_bin_table = sparse.csr_matrix(req_vs_collid_bin_table)
#    print("req_vs_collid_bin_table  ",  req_vs_collid_bin_table)
    print("End of variables \n")


    print("Beginning of constraints \n")
    # Add flow conservation (path selection) constraints 
    # Sum_{p in P, w in W} y_rpw <= v_r=1, r in R
    path_sel_ctr = sparse.kron(sparse.eye(R),np.ones((1,P*W)),format = 'lil')
    path_sel_senses = ["L"] * R
    path_sel_rhs = [1.0] * R  
    
    print("Adding flow conserv constraints \n")
#    ctr_matrix = sparse.lil_matrix(path_sel_ctr)
#    indeces_list = ctr_matrix.rows.tolist()
#    data_list = ctr_matrix.data.tolist()
#    lin_expr = [[indeces_list[i], data_list[i]] for i in range(len(data_list))]  
    # a list of SparsePair instances or a matrix in list-of-lists format.
    ind = path_sel_ctr.rows
    val = path_sel_ctr.data
#   lin_expr =list(zip(ind, val))  
#    lin_expr = list(it.zip_longest(ind, val))
    del path_sel_ctr
#    gc.collect()
#    print(iv)
#    print(type(iv))
#    print(len(iv))
    #lin_expr = iv
    #lin_expr = [cplex.SparsePair(ind,val)]
    #print("lin_expr",lin_expr)
    #print("type lin_expr", type(lin_expr))

    #lin_expr = [cplex.SparsePair(ind[i],val[i]) for i in range(R)] 
    #lin_expr = np.array([path_sel_ctr.rows, path_sel_ctr.data]).transpose().tolist()    
    model.linear_constraints.add(lin_expr = list(it.zip_longest(ind, val)), senses = path_sel_senses, rhs = path_sel_rhs)
    del ind, val, path_sel_senses, path_sel_rhs
#    gc.collect()
    
    # Sum_{c in R_r, p in P_l} y_cpw <= 1, r in r, l in L, w in W
    wv_clash_ctr = sparse.kron(req_vs_collid_bin_table, sparse.kron(link_vs_path_bin_table, sparse.eye(W)),format = 'lil')
    wv_clash_senses = ["L"]*L*W
    wv_clash_rhs = [1.0] *L*W   
#    wv_clash_senses = ["L"]* R*L*W
#    wv_clash_rhs = [1.0] * R*L*W     
    print("Adding capacity constraints \n")
    ind = wv_clash_ctr.rows
    val = wv_clash_ctr.data
    print("capacity ind val extracted \n")
#    lin_expr = list(it.zip_longest(ind, val))
#    lin_expr =list(zip(ind, val))
#    print("capacity lin_expr created \n")
    del wv_clash_ctr
#    gc.collect()

    for r in range(R):
        print("r= ",r)
##        print("creating lin_expr \n")
#        #lin_expr =list(zip(ind[r*L*W : (r+1)*L*W],val[r*L*W : (r+1)*L*W]))
##        #wv_clash_ctr_this_req = wv_clash_ctr[r*L*W : (r+1)*L*W,:]
###        ind_lc = ind[r*L*W : (r+1)*L*W]
###        val_lc = val[r*L*W : (r+1)*L*W]
###        lin_expr = [cplex.SparsePair(ind_lc[i],val_lc[i]) for i in range(L*W)] 
##        lin_expr = list(zip(ind, val))
##        #lin_expr = [cplex.SparsePair(ind[r*L*W+i],val[r*L*W+i]) for i in range(L*W)] 
##        #lin_expr = np.array([ind[r*L*W : (r+1)*L*W] , val[r*L*W : (r+1)*L*W]]).transpose().tolist()  
#        print("adding lin_expr \n")
        
        print("adding lin_expr \n")    
        # a list of SparsePair instances or a matrix in list-of-lists format.
        model.linear_constraints.add(lin_expr = list(it.zip_longest(ind[r*L*W : (r+1)*L*W] , val[r*L*W : (r+1)*L*W])), senses = wv_clash_senses, rhs = wv_clash_rhs)
        
#        #model.linear_constraints.add(lin_expr, senses = wv_clash_senses, rhs = wv_clash_rhs)
#        model.linear_constraints.add(lin_expr[r*L*W : (r+1)*L*W], senses = wv_clash_senses, rhs = wv_clash_rhs)
#    del lin_expr, wv_clash_senses, wv_clash_rhs
#    gc.collect()
               
    #ctr_matrix = sparse.lil_matrix(wv_clash_ctr)
#    indeces_list = ctr_matrix.rows.tolist()
#    data_list = ctr_matrix.data.tolist()
#    print("creating lin_expr \n")
#    lin_expr = [[indeces_list[i], data_list[i]] for i in range(len(data_list))] 
        
#    lin_expr = np.array([wv_clash_ctr.rows, wv_clash_ctr.data]).transpose().tolist()  
    
#    print("creating lin_expr \n")
#    ind = wv_clash_ctr.rows
#    val = wv_clash_ctr.data
#    lin_expr = [cplex.SparsePair(ind[i],val[i]) for i in range(R*L*W)]  
           
#    print("adding lin_expr \n")    
    # a list of SparsePair instances or a matrix in list-of-lists format.
#    model.linear_constraints.add(lin_expr = list(it.zip_longest(ind, val)), senses = wv_clash_senses, rhs = wv_clash_rhs)
    del ind, val, wv_clash_senses, wv_clash_rhs
#    gc.collect()
    
#    # we group everything
#    senses = path_sel_senses + wv_clash_senses
#    rhs = path_sel_rhs + wv_clash_rhs
#    ctr_matrix = sparse.lil_matrix(sparse.vstack([path_sel_ctr , wv_clash_ctr],format="lil"))
#    indeces_list = ctr_matrix.rows.tolist()
#    data_list = ctr_matrix.data.tolist()
#   # lin_expr = [ctr_matrix.rows.tolist(), ctr_matrix.data.tolist()]
#   
#    print("Adding constraints \n")
#    lin_expr = [[indeces_list[i], data_list[i]] for i in range(len(data_list))]  
#    # a list of SparsePair instances or a matrix in list-of-lists format.
#    model.linear_constraints.add(lin_expr, senses, rhs)
    print("End of constraints \n")

    # Export the model to disk and solve    
    print("Starting model \n")
    
    try:
        #model.write("models/mod_%s.lp"%strfile)
        model.solve()
    except CplexSolverError as e:
        print("Exception raised during solve: " + e)
    else:
        # Solve succesful, dump solution.
        code_status = model.solution.get_status()
        str_status = model.solution.status[code_status]
        print("Solution status = ", code_status)
        print("Solution status msg = ", str_status)\
        
        if code_status==103:#infeasible
            np.savez("optimValues/opt_%s"%strfile, code_status = code_status, str_status = str_status)
        else :
            objective = model.solution.get_objective_value()       
            print("Objective = " + str(objective))

            #save solution
            #model.solution.write("solutions/sol_%s.sol"%strfile)
            x_rpw_opt = np.asarray(model.solution.get_values())
#           x_rpw_opt = np.asarray(model.solution.get_values(indices_x_rpw[0],indices_x_rpw[-1]))
 #           x_trpw_opt = np.asarray(model.solution.get_values(indices_x_trpw[0],indices_x_trpw[-1]))    
    
    print("Processing opt sol \n")
    wv_clash_ctr = sparse.kron(req_vs_collid_bin_table, sparse.kron(link_vs_path_bin_table, sparse.eye(W)))
    x_rlw_opt = wv_clash_ctr.dot(x_rpw_opt.reshape(num_x_rpw,1))
    del wv_clash_ctr
    gc.collect()
    x_r_lw_opt = x_rlw_opt.reshape((R,L*W))
    x_data[:,0:L*W] = 1 - req_vs_collid_bin_table.transpose().dot(x_r_lw_opt) #x_data is free capacity
    x_data[:,L*W] = reqSrcs
    x_data[:,L*W+1] = reqDsts
    x_data[:,L*W+2] = 1

    x_r_pw_opt = x_rpw_opt.reshape((R,P*W)) 
    opt_r_ids, opt_pw_ids = np.nonzero(x_r_pw_opt)
    blocked_r_ids = np.setdiff1d(np.arange(R), opt_r_ids)
#    print('opt_r_ids' , opt_r_ids)
#    print('blocked_r_ids' , blocked_r_ids)
#    print('opt_pw_ids' , opt_pw_ids)
    p,w = np.divmod(opt_pw_ids, W)
    sd,k = np.divmod(p, numPaths_per_nodepair)
#    print('k' , k)
#    print('w' , w)
#    print('kw' , k*W+w)
    y_data[opt_r_ids,1+ k*W+w] = 1
    if blocked_r_ids.size > 0:
        y_data[blocked_r_ids,0] = 1
#    np.savez("optimValues/opt_%s"%strfile, code_status = code_status, str_status = str_status, objective=objective
#     , a_i_opt=a_i_opt,z_i_opt=z_i_opt, z_ie_opt= z_ie_opt, z_ieur_opt=z_ieur_opt, x_eur_opt=x_eur_opt, x_eurij_opt=x_eurij_opt, y_pw_opt=y_pw_opt)    
    
    print('\n totalNumLPs: ', R)
    for local_pathId in range(numPaths_per_nodepair) :
        print('estNumLPs of path ', local_pathId, ': ', np.sum(y_data[:,1 + local_pathId*W:1+(local_pathId+1)*W ]) )    
    print('blockedNumLPs: ', np.sum(y_data[:,0]))
    
    dataset = {}
    dataset['X'] = x_data.tolist()
    dataset['y'] = y_data.tolist()
    dataset['topology'] = y_data.tolist()

    full_filename = sol_pathname + "ILP_oracle.json"
    if not os.path.exists(sol_pathname):
        os.makedirs(sol_pathname)
    with open(full_filename, 'w') as outfile:
        print("Done : " , full_filename)
        json.dump(dataset, outfile)  
        outfile.close()