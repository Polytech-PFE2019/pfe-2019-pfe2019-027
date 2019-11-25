#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:05:49 2019

@author: raparicio
"""
import sys
sys.path.insert(0, "/Applications/CPLEX_Studio128/cplex/python/3.6/x86-64_osx")
import cplex
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
import os
#import time
import numpy as np
from scipy import sparse
np.set_printoptions(threshold=np.inf)
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
import json

def ILP_oracle_slow(sol_pathname,topology, arrivalsList, numWavelengths, strfile):
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
    #lastDeparture = np.amax(reqDepartTimes)*0.5
    #print(reqArrivTimes)
    #print(reqDepartTimes)
    #print(lastDeparture)
    #print(reqDepartTimes[-1])
    lastDeparture = reqDepartTimes[-1]
    interArrivTimes = np.append(reqArrivTimes[1:],lastDeparture) - reqArrivTimes
    reqSDs = N*reqSrcs + reqDsts
    req_vs_path_bin_table = sparse.csr_matrix(sd_vs_path_bin_table[reqSDs,:])
        
    y_data = np.zeros((R, 1+ numPaths_per_nodepair*W)) # first position [0] is no_solution
#    y_linkBased_data = np.zeros((numReqs, numLinks*numWavelengths)) # 1 if req used lw, o otherwise   
    x_data = np.zeros((R, L*W + 3)) # 1 if lw is free  
#    link_wavelength_state = np.zeros((numLinks, numWavelengths)) # 1 if (l,w) is used
     
    # Create a new (empty) model and populate it below.
    model = cplex.Cplex()
    model.objective.set_sense(model.objective.sense.maximize)

    print("Beginning of variables \n")
    # Add the y_rpw variables
    num_y_rpw = R*P*W
    #obj = np.kron(reqDurations, np.ones((P*W))).tolist()
    obj = [1.0] * num_y_rpw
    lb = [0.0] * num_y_rpw
    ub_y_rpw = sparse.kron(req_vs_path_bin_table.reshape((R*P,1)), np.ones((W,1)) ) 
    ub = ub_y_rpw.toarray().flatten().tolist()
    #ub = [1.0] * num_y_rpw
    types = ["B"]*num_y_rpw 
    # set up names of variables
    names = ["y_%s_%s_%s" % (r,p,w) for r in range(R) for p in range(P) for w in range(W)] 
    # add variables to the model and store indices as a list        
    indices_y_rpw = list(model.variables.add(obj, lb, ub, types, names))
    
    # Add the x_trpw variables
    num_x_trpw = R*R*P*W
    obj = [0.0] * num_x_trpw 
    lb = [0.0] * num_x_trpw
    ub_x_trpw = sparse.csr_matrix((R,R*P*W))
    #ub_x_trpw = sparse.csr_matrix((R,R*P*W))
    #sparse.kron( ones((1, R)) , ub_y_rpw)) #[1.0] * num_x_trpw
    for t in range(R):
#        aux = sparse.csr_matrix((1,R*P))
#        print("reqArrivTimes ", reqArrivTimes)
#        print("reqDepartTimes  ",  reqDepartTimes)
#        livingReqs1 = np.nonzero((reqArrivTimes <= reqArrivTimes[t]))
#        livingReqs2 = np.nonzero((reqArrivTimes[t] <= reqDepartTimes))
        livingReqs = np.nonzero((reqArrivTimes <= reqArrivTimes[t]) & (reqArrivTimes[t] <= reqDepartTimes))[0]
        aux_r = sparse.lil_matrix((1,R))
        aux_r[:,livingReqs] = 1
        ub_x_trpw[t,:] = sparse.kron(aux_r , np.ones((1,P*W)))

#        print("livingReqs1 for ", t, " " , livingReqs1)
#        print("livingReqs2 for ", t, " " , livingReqs2)
#        print("livingReqs3 for ", t, " " , livingReqs)

#        aux[livingReqs,:] = 1
#        print("aux[livingReqs,:] ", aux[livingReqs,:].todense())
#        aux[livingReqs,:] = req_vs_path_bin_table[livingReqs,:]

        #aux[(reqArrivTimes <= reqArrivTimes[t]) & (reqArrivTimes[t] <= reqDepartTimes),:] = req_vs_path_bin_table[(reqArrivTimes <= reqArrivTimes[t]) & (reqArrivTimes[t] <= reqDepartTimes),:]
#        ub_x_trpw[t,:] = sparse.kron(aux.reshape((1,R*P)) , np.ones((1, W))) # 1x RPW slice 
#        print("2 aux[livingReqs,:] ", aux[livingReqs,:].todense())
                    
    ub = ub_x_trpw.reshape((num_x_trpw,1)).toarray().flatten().tolist()
    #ub = [1.0] * num_x_trpw
    types = ["B"]*num_x_trpw 
    # set up names of variables
    names = ["x_%s_%s_%s_%s" % (t,r,p,w) for t in range(R) for r in range(R) for p in range(P) for w in range(W)] 
    # add variables to the model and store indices as a list
    indices_x_trpw = list(model.variables.add(obj, lb, ub, types, names))    
    print("End of variables \n")

    # Add flow conservation (path selection) constraints 
    # Sum_{p in P, w in W} y_rpw <= v_r=1, r in R
    y_coeffs = sparse.kron(sparse.eye(R),np.ones((1,P*W)))
    x_coeffs = sparse.csr_matrix((R,num_x_trpw))
    path_sel_ctr = sparse.hstack([y_coeffs , x_coeffs],format="csr") 
    path_sel_senses = ["L"] * R
    path_sel_rhs = [1.0] * R  
    
    # Sum_{r in R, p in P_l} x^t_rpw <= 1, t in T, l in L, w in W
    y_coeffs = sparse.csr_matrix((R*L*W, num_y_rpw))
    x_coeffs = sparse.kron(sparse.eye(R) , sparse.kron(np.ones((1,R)) , sparse.kron(sparse.csr_matrix(link_vs_path_bin_table), sparse.eye(W))))
    wv_clash_ctr = sparse.hstack([y_coeffs , x_coeffs],format="csr") 
    wv_clash_senses = ["L"]* R*L*W
    wv_clash_rhs = [1.0] * R*L*W      
    
    #  d_r*y_rpw - Sum_{t in T} iat_t x^t_rpw <= 0, (r,p,w), 
    #t such that a_r <= a_t <= a_r+d_r, (r,p,w)
    y_coeffs = sparse.kron(sparse.diags(reqDurations, 0), sparse.eye(P*W))
    x_coeffs = sparse.kron(interArrivTimes.reshape((1,R)), sparse.eye(R*P*W))
    bind_ctr = sparse.hstack([y_coeffs , -x_coeffs],format="csr") 
    bind_senses = ["L"]* R*P*W
    bind_rhs = [0.0] * R*P*W  
    
    # we group everything
    senses = path_sel_senses + wv_clash_senses + bind_senses
    rhs = path_sel_rhs + wv_clash_rhs + bind_rhs
    ctr_matrix = sparse.lil_matrix(sparse.vstack([path_sel_ctr , wv_clash_ctr , bind_ctr],format="lil"))
    indeces_list = ctr_matrix.rows.tolist()
    data_list = ctr_matrix.data.tolist()
   # lin_expr = [ctr_matrix.rows.tolist(), ctr_matrix.data.tolist()]
   
    print("Adding constraints \n")
    lin_expr = [[indeces_list[i], data_list[i]] for i in range(len(data_list))]  
    # a list of SparsePair instances or a matrix in list-of-lists format.
    model.linear_constraints.add(lin_expr, senses, rhs)
    print("End of constraints \n")

#    constraints = list()
#for i in range(size):
#    constraints.append([nmindices[i * size : (i + 1) * size], [1.0] * size])
#for i in range(size):
#    constraints.append(cplex.SparsePair(nmindices[i : size * size : size], [1.0] * size))
#
#    model.linear_constraints.add(lin_expr = constraints, senses = constraint_senses, rhs = rhs)
#    model.linear_constraints.add(lin_expr=[cplex.SparsePair(indices,values)], senses="L", rhs)

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
            y_rpw_opt = np.asarray(model.solution.get_values(indices_y_rpw[0],indices_y_rpw[-1]))
            x_trpw_opt = np.asarray(model.solution.get_values(indices_x_trpw[0],indices_x_trpw[-1]))    
    
    print("Processing opt sol \n")

    x_tlw_opt = wv_clash_ctr[:,-num_x_trpw:].dot(x_trpw_opt.reshape(num_x_trpw,1))
    x_data[:,0:L*W] = 1 - x_tlw_opt.reshape((R,L*W)) #x_data is free capacity
    x_data[:,L*W] = reqSrcs
    x_data[:,L*W+1] = reqDsts
    x_data[:,L*W+2] = 1

    y_r_pw_opt = y_rpw_opt.reshape((R,P*W)) 
    opt_r_ids, opt_pw_ids = np.nonzero(y_r_pw_opt)
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

    full_filename = sol_pathname + "ILP_oracle_slow.json"
    if not os.path.exists(sol_pathname):
        os.makedirs(sol_pathname)
    with open(full_filename, 'w') as outfile:
        print("Done : " , full_filename)
        json.dump(dataset, outfile)  
        outfile.close()