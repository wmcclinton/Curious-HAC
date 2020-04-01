import pickle
import matplotlib.pyplot as plt
import numpy as np

N = 100

not_curious_data = pickle.load(open("MiniGrid-DoorKey-8x8-v0_ppo_seed1_20-04-01-01-52-39.p", "rb"))
curious_data = pickle.load(open("MiniGrid-DoorKey-8x8-v0_ppo_seed1_20-04-01-03-39-48.p", "rb"))

def get_episode_rewards(data):
    intrinsic_rewards = []
    extrinsic_rewards = []

    eps_r_i = []
    eps_r_e = []

    for datum in data:
        eps_r_i.append(datum[0])
        eps_r_e.append(datum[1])
        if datum[2]:
            intrinsic_rewards.append(sum(eps_r_i))
            extrinsic_rewards.append(sum(eps_r_e))
            eps_r_i = []
            eps_r_e = []

    return intrinsic_rewards, extrinsic_rewards

n_data_r_i, n_data_r_e = get_episode_rewards(not_curious_data)
c_data_r_i, c_data_r_e = get_episode_rewards(curious_data)

plt.plot(np.convolve(n_data_r_e, np.ones((N,))/N, mode='valid'),label='PPO',color='r')
plt.plot(np.convolve(c_data_r_e, np.ones((N,))/N, mode='valid'),label='Curious+PPO',color='b')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('avg_extrinsic_reward')
plt.xlabel('episodes')
plt.show()