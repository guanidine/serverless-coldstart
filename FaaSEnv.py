import gym
from gym import spaces

import generator


class FaaSEnv(gym.Env):
    def __init__(self, max_num=100):
        self.max_num = max_num

        self.preds = [0] * 8
        self.reals = [0] * 8

        self.action_space = spaces.Box(low=0, high=self.max_num, shape=(1,), dtype=float)
        self.observation_space = spaces.Box(low=0, high=self.max_num, shape=(16,), dtype=float)

        t, f = generator.generate()

        # self.action_space = spaces.Discrete(5)
        # self.observation_space = spaces.Discrete(2)

        self.seed()

    def step(self, action):

        state = 1

        if action == 2:
            reward = 1
        else:
            reward = -1

        done = True
        info = {}
        return state, reward, done, info

    def reset(self):
        state = 0
        return state

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env

    env = FaaSEnv()
    check_env(env)
