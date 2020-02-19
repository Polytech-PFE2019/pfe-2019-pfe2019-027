import matplotlib.pyplot as plt
import json
import numpy as np

def display_stats(nnode, ndeg,f_reward):

    data_dqn = json.load(open(f"results/n{nnode}_d{ndeg}_r{f_reward}_DQN.json"))
    data_rwa = json.load(open(f"results/n{nnode}_d{ndeg}_RWA.json"))

    r_dqn, a_dqn,  = np.array(data_dqn["results"]), np.array(data_dqn["actions"])
    lr = r_dqn.shape[0]
    print(lr)
    nr = np.sum(r_dqn[0,:])
    r_rwa, a_rwa = np.array(data_rwa["results"]), np.array(data_rwa["actions"])
    r_rwa, a_rwa = r_rwa[:lr,:], a_rwa[:lr,:]
    r_dqn, a_dqn, r_rwa, a_rwa = r_dqn/nr, a_dqn/nr, r_rwa/nr, a_rwa/nr

    fig, ax = plt.subplots(1,2, figsize=(10,10))

    labels = ["Accepted", "Blocked", "Skipped"]

    ax[0].set_title("Requests statutes")
    for i in range(3):
        ax[0].plot(np.arange(r_dqn.shape[0]), r_dqn[:,i])
    ax[0].legend(labels, loc="lower left")


    # ax[0].set_title("Greedy results")
    for i in range(3):
        ax[0].scatter(np.arange(r_rwa.shape[0]), r_rwa[:,i],s=1)
    # ax[0].legend(labels)

    labels = list(map(str, range(4)))
    ax[1].set_title("Chosen actions")
    for i in range(4):
        ax[1].plot(np.arange(a_dqn.shape[0]), a_dqn[:,i])
    ax[1].legend(labels)


    # ax[1].set_title("Greedy Actions")
    for i in range(4):
        ax[1].scatter(np.arange(a_rwa.shape[0]), a_rwa[:,i],s=1)
    # ax[1].legend(labels)


    plt.tight_layout()
    plt.show()

n=10
d=5
r=1

display_stats(n,d,2)
