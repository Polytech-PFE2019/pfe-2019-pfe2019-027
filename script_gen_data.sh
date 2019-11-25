#!/usr/bin/env bash

python3 main_SL.py --state 1 --which_nn_use SL_FC_LinkToPath 
python3 main_SL.py --state 1 --which_nn_use NANE_LinkToPath 


#python3 main_SL.py --state 0 --which_nn_use SL_FC_LinkToPath --epochs 50
#python3 main_SL.py --state 0 --which_nn_use NANE_LinkToPath --epochs 50
#python3 main_SL.py --state 1 --which_nn_use SL_FC_LinkToPath --epochs 50
#python3 main_SL.py --state 1 --which_nn_use NANE_LinkToPath --epochs 50



#python3 generateDatasetMain.py --state 0 --network_node 10 --link_capacity 1 --Total_R 10000 --k_path_generated 5 --nb_random_graph 10 
#python3 generateDatasetMain.py --state 1 --network_node 10 --link_capacity 1 --Total_R 10000 --k_path_generated 5 --nb_random_graph 10 
#python3 generateDatasetMain.py --state 2 --network_node 10 --link_capacity 1 --Total_R 10000 --k_path_generated 5 --nb_random_graph 10
 


#python3 main.py --state 0 --which_nn_use SL_FC_LinkToPath --epochs 50 --k_path_generated 5
#python3 main.py --state 1 --which_nn_use SL_FC_LinkToPath --epochs 50 --k_path_generated 5
#python3 main.py --state 2 --which_nn_use SL_FC_LinkToPath --epochs 50 --k_path_generated 5

#python3 main.py --state 2 --which_nn_use SL_FC_LinkToPath --activation sigmoid  --epochs 50 --k_path_generated 5

#python3 main.py --state 0 --which_nn_use NANE_LinkToPath --epochs 50 --k_path_generated 5
#python3 main.py --state 1 --which_nn_use NANE_LinkToPath --epochs 50 --k_path_generated 5
#python3 main.py --state 2 --which_nn_use NANE_LinkToPath --epochs 50 --k_path_generated 5


#python3 main.py --state 2 --which_nn_use NANE_LinkToPath --activation sigmoid --epochs 50 --k_path_generated 5



#python3 generateDatasetMain.py --network_node 10  --link_capacity 1 --GenerateNetworkSample --Total_R 10000 --k_path_generated 5 --nb_random_graph 10

 