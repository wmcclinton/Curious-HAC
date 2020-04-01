import gym
import gym_minigrid

from gym import RewardWrapper

import gym.spaces as spaces
from gym import ObservationWrapper

def f(reward):
    return 0

class TransformReward(RewardWrapper):
    def __init__(self, env, f):
        super(TransformReward, self).__init__(env)
        assert callable(f)
        self.f = f

    def reward(self, reward):
        return self.f(reward)

class XYObservation(ObservationWrapper):
    def __init__(self, env):
        super(XYObservation, self).__init__(env)
        self.env = env

    def observation(self, observation):
        observation["pos"] = [[self.env.agent_pos[0], self.env.agent_pos[1]]]
        # add whether the agent has the key to pos
        return observation

def make_env(env_key, seed=None):
    env = gym.make(env_key)
    #env = TransformReward(env,f)
    #env = FullyObsWrapper(env)
    env = XYObservation(env)

    env.seed(seed)
    return env
