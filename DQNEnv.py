import numpy as np
from gym import Env, spaces
import json
import os
from collections import namedtuple
import matplotlib.pyplot as plt

from utils.trace_loader import TraceLoader

class DQNEnv(Env):

    def __init__(self, k_path_generated=3, gen_state=1, numWavelengths=4,
    nodes=[5], degree=[2], netw_i=0, lpMat_i=0, maxUtilPerNodePair=[0.3],
    reward_function = "basic"):

        super(DQNEnv, self).__init__()

        self.traces = self.load_traces(nodes, lpMat_i, maxUtilPerNodePair)
        self.trace_number = len(self.traces)
        self.numReqs = len(self.traces[0])

        self.topology = self.load_topology(nodes, degree, netw_i)

        self.sd_vs_path_bin_table = np.array(self.topology['sd_vs_path_bin_table'])
        self.link_vs_path_bin_table = np.array(self.topology['link_vs_path_bin_table'])
        self.node_vs_path_bin_table = np.array(self.topology['node_vs_path_bin_table'])
        self.links = np.array(self.topology["links"])

        self.numNodes = self.node_vs_path_bin_table.shape[0]
        self.numLinks = self.link_vs_path_bin_table.shape[0]
        self.numPaths_per_nodepair = k_path_generated
        self.numWavelengths = numWavelengths

        # self.y_data = np.zeros((self.numReqs, 1+ self.numPaths_per_nodepair*self.numWavelengths))
        self.y_linkBased_data = np.zeros((self.numReqs, self.numLinks*self.numWavelengths))
        # slef.x_data = np.zeros((self.numReqs, self.numLinks*self.numWavelengths + 3))
        self.link_wavelength_state = np.zeros((self.numLinks, self.numWavelengths))

        if (gen_state == 0):
            self.state_size = self.numLinks*self.numWavelengths + self.numNodes
        else:
            self.state_size = (1 + self.numLinks*self.numWavelengths) * self.numNodes

        self.action_space = spaces.Discrete(k_path_generated+1)
        # self.observation_space = spaces.Discrete(self.state_size)
        self.observation_space = spaces.Box(0,1,[self.numLinks, self.numWavelengths + 2])

        self.reward_function = reward_function

        self.previous_action = None

        self.next_trace = 0
        self.n_request = 0
        self.actual_trace = list()
        self.actual_request = list()
        self.ongoing_requests = np.zeros((0,2))

        #Statistics values

        self.epoch_results = [0,0,0] #[accepted_requests,blocked_requests,skipped_requests]
        self.global_results = []
        self.epoch_actions = [0] * (1+self.numPaths_per_nodepair)
        self.global_actions = []

    def reset(self):
        self.actual_trace = self.traces[self.next_trace%self.trace_number]
        self.next_trace += 1
        self.n_request = 0
        self.actual_request = self.actual_trace[self.n_request,:]

        # self.y_data = np.zeros((self.numReqs, 1+ self.numPaths_per_nodepair*self.numWavelengths))
        self.y_linkBased_data = np.zeros((self.numReqs, self.numLinks*self.numWavelengths))
        # slef.x_data = np.zeros((self.numReqs, self.numLinks*self.numWavelengths + 3))
        self.link_wavelength_state = np.zeros((self.numLinks, self.numWavelengths))

        if sum(self.epoch_results)!=0:
            self.global_results.append(self.epoch_results)
            self.epoch_results = [0,0,0]

        if sum(self.epoch_actions)!=0:
            self.global_actions.append(self.epoch_actions)
            self.epoch_actions = [0,0,0,0]

        self.ongoing_requests = np.zeros((0,2))

        print(f"\rEpoch {self.next_trace}", end="")

        return self.get_obs()

    def step(self, action):

        self.previous_action = action

        n1,n2,t,d = self.actual_request

        #Free previous path for which the according request is finished
        self.free_available_path(t)

        sd = int(n1 * self.numNodes + n2)
        if action:
            valid = False

            chosen_path = action - 1

            #Retrieve available paths ids from n1 to n2
            precalculated_paths = np.nonzero(self.sd_vs_path_bin_table[sd])[0]
            #If there is less than k possible path and the action is above the number of path, consider it as invalid response
            if chosen_path < precalculated_paths.size:
                #Get te id of the path
                id_path = precalculated_paths[chosen_path]
                #Retrieve the links included in the chosen path
                pathLinkIds = np.nonzero(self.link_vs_path_bin_table[:,id_path])[0]
                #Check if there is a Wavelength for which the path is free
                for wavelengthId in range(self.numWavelengths):
                    if np.sum(self.link_wavelength_state[pathLinkIds,wavelengthId]) == 0 :
                        #If a path is free, update matrices
                        self.y_linkBased_data[self.n_request, pathLinkIds*self.numWavelengths + wavelengthId] = 1
                        self.link_wavelength_state[pathLinkIds, wavelengthId] = 1

                        #Adding the request and the links used to the history
                        self.ongoing_requests = np.append(self.ongoing_requests, [[self.n_request, t+d]], axis=0)

                        valid = True
                        break

        else:
            valid = True
            #check if a path was possible but not taken
            candPathIds = np.nonzero(self.sd_vs_path_bin_table[sd])[0]
            numCandPaths_per_nodepair = candPathIds.size
            for local_pathId in range(numCandPaths_per_nodepair): #0....K
                pathId = candPathIds[local_pathId]
                pathLinkIds = np.nonzero(self.link_vs_path_bin_table[:,pathId])[0]
                for wavelengthId in range(self.numWavelengths):
                    if np.sum(self.link_wavelength_state[pathLinkIds,wavelengthId]) == 0 :
                        valid = False
                        break
                if not valid:
                    break

        self.update_stats(action, valid)
        reward = self.reward(action, self.actual_request, valid)

        self.n_request += 1
        done = self.n_request >= self.actual_trace.shape[0]

        self.actual_request = None if done else self.actual_trace[self.n_request,:]

        obs = self.get_obs(done)
        info = self.get_info(valid)

        return obs, reward, done, info


    def reward(self, action, request, valid):

        def basic_function(action, request, valid):
            return (1 if valid else -1)

        def no_zero_penalty(action, request, valid):
            if valid:
                return 1
            else:
                if action:
                    return -1
                else:
                    return 0

        def refined_no_zero_penalty(action, request, valid):
            if valid:
                return 1
            else:
                if action:
                    return -1
                else:
                    return 0

        def delay_function(action, request, valid):
            if action == 0:
                return 0 if valid else -1
            else:
                return (1 if valid else -1) * (request[3])

        def inverse_delay_function(action, request, valid):
            if action == 0:
                return 0 if valid else -1
            else:
                return (1 if valid else -1) * (1/request[3])

        transcript = {
            1 : basic_function,
            2 : no_zero_penalty,
            3 : refined_no_zero_penalty,
            4 : delay_function,
            5 : inverse_delay_function
        }

        return transcript[self.reward_function](action, request, valid)

    def update_stats(self,action, valid):
        self.epoch_actions[action]+=1

        if action:
            if valid:
                self.epoch_results[0]+=1
            else:
                self.epoch_results[2]+=1
        else:
            self.epoch_results[1]+=1

    def get_path(self, start_node, end_node, action):
        path_idx = start_node * (self.numNodes-1) + end_node - (1 if end_node > start_node else 0)
        return self.topology["list_paths"][path_idx][action - 1]

    def get_obs(self, done=False):

        if done:
            return np.zeros((self.numLinks, self.numWavelengths + 2))

        s,e,t,d = self.actual_request

        #Add two columns to the link_wavelength_state path_matrix
        #First one to specify links starting from the start of the next request
        #Second one to specify links reaching the target of the next request
        start_links=np.argwhere(self.links[:,0] == s)
        target_links=np.argwhere(self.links[:,1] == e)

        start_matrix = np.zeros((self.numLinks,1))
        target_matrix = np.zeros((self.numLinks,1))
        start_matrix[start_links,:]=d
        target_matrix[target_links,:]=-d

        req_info = np.hstack((start_matrix,target_matrix))
        obs = np.hstack((self.link_wavelength_state,req_info))

        return obs

    def get_info(self, valid):
        return {
            "Number_of_accepted_skipped": self.epoch_results[0],
             "Number_of_blocked_response": self.epoch_results[1],
             "Number_of_skipped_response": self.epoch_results[2],
             "is_success": valid
             }

    def free_available_path(self, arriv_time):
        departingReqs_ind = np.argwhere(self.ongoing_requests[:,1] < arriv_time)
        numDepartingReqs = departingReqs_ind.size
        if numDepartingReqs > 0:
            requestsToRelease = self.ongoing_requests[departingReqs_ind ,0]
            for reqToRelease_idx in range(numDepartingReqs):
                reqToRelease = int(requestsToRelease[reqToRelease_idx])
                linkWavelengthsToRelease = np.nonzero(self.y_linkBased_data[reqToRelease,:])[0]
                linkToRel,wavelToRel = np.divmod(linkWavelengthsToRelease, self.numWavelengths)
                self.link_wavelength_state[linkToRel, wavelToRel[0]] = 0
            self.ongoing_requests = np.delete(self.ongoing_requests, departingReqs_ind, axis=0)

    def render(self):
        pass


    def show_params(self):
        print("Topology parameters:")
        print(f"Number of Nodes: {self.numNodes}")
        print(f"Number of Links: {self.numLinks}")
        print(f"Number of Wavelength: {self.numWavelengths}")

    def load_traces(self, nodes=[10], lpMat_i=0, maxUtilPerNodePair=[0.3]):
        traces_path = f"Data_set/raw_data/{nodes[0]}node/{lpMat_i}_traff_matrix/{int(maxUtilPerNodePair[0]*100)}_ut/"
        return TraceLoader(traces_path)

    def load_topology(self, nodes=[10], degree=[5], netw_i=0):
        topology_path = f"Data_set/raw_data/{nodes[0]}node/{degree[0]}degree/{netw_i}_instance/"
        return json.load(open(topology_path+"topology.json"))

    def get_history(self):
        return {"results": self.global_results, "actions": self.global_actions }
