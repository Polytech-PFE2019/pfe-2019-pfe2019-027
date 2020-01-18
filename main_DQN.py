import os
import json
import numpy as np
import random
import argparse
import shutil

import matplotlib.pyplot as plt

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
#from NANE_Model import NanePolicy
from stable_baselines import DQN

from DQNEnv import DQNEnv

ap=argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="MLP_FC")
ap.add_argument("-k", "--kpath", default=3)
ap.add_argument("-n", "--nodes", default=10)
ap.add_argument("-d", "--degree", default=5)
ap.add_argument("-w", "--wavelength", default=4)
ap.add_argument("-c", "--clearlogs", default=True)
ap.add_argument("-r", "--rewardfunction", default="basic")
ap.add_argument("-l", "--pielogs", default=False)
args = vars(ap.parse_args())

def print_info(info):
    for a,b in info.items():
        if a=="is_success": continue
        t=a.replace("_"," ")
        print(f"{t}: {b}")

def render_info(info):
    labels = [i for i in info if i!="is_success"]
    data = [info[i] for i in labels]
    labels = [x.replace("_", " ") for x in labels]

    for a,b in zip(labels, data):
        print(f"{a}: {b}")

    explode = tuple(0.1 if i==max(data) else 0 for i in data)

    #add colors
    colors = ['#ff9999','#66b3ff','#99ff99']
    fig1, ax1 = plt.subplots()
    ax1.pie(data, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()
    plt.show()


def main():
    log_folder="./logs"
    #If log folder exist
    if os.path.isdir(log_folder):
        #Clean it
        if args["clearlogs"] != "False":
            print("Clearing logs")
            for filename in os.listdir(log_folder):
                shutil.rmtree("/".join([log_folder,filename]))
    else:
        #Otherwise, create it
        print("Creating Log folder")
        os.mkdir(log_folder)

    nodes = [args["nodes"]]
    degree = [args["degree"]]
    netw_i = 0
    lpMat_i = 0
    maxUtilPerNodePair = [0.3]
    nbNetwork = 10
    traffic="uniform"

    k_path_generated = args["kpath"]
    gen_state = 1
    numWavelengths = args["wavelength"]

    reward_function = args["rewardfunction"]

    env = DQNEnv(
        nodes=nodes,
        degree=degree,
        netw_i=netw_i,
        lpMat_i=lpMat_i,
        maxUtilPerNodePair=maxUtilPerNodePair,
        k_path_generated=k_path_generated,
        gen_state=gen_state,
        numWavelengths=numWavelengths,
        reward_function = reward_function
    )

    env.show_params()

    model_name=args["model"]
    if model_name=="NANE":
        print("Using NANE Layers")
        #model_architecture = NanePolicy
    else:
        print("Using Fully connected layers")
        model_architecture = MlpPolicy

    model = DQN(model_architecture, env, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=25000)

    obs = env.reset()

    done=False

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        env.render()

    print("\nTest results:")
    log_function = render_info if args["pielogs"]=="True" else print_info
    log_function(info)

main()