import gym
import numpy as np
import torch
from gym import spaces
from torch import optim

import config
import generator
from lstm import LSTM
from utils import load_checkpoint, normalize, cal_reward


class FaaSEnv(gym.Env):
    def __init__(self, max_num=100):
        self.max_num = max_num

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 8), dtype=float)

        self.state = np.zeros((3, 8))

        _, self.f = generator.generate()
        self.f = normalize(self.f)
        self.t = 0

        self.model = LSTM(device=config.DEVICE).to(config.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LSTM_LEARNING_RATE)

        self.init_obs()

        self.seed()

    def step(self, action):
        self.state[0] = np.append(self.state[0][1:], self.f[self.t].item())
        self.state[1] = np.append(self.state[1][1:], self.state[2][0])
        self.state[2][0] = self.model(
            torch.from_numpy(self.state[0].astype(np.float32)).to(config.DEVICE).unsqueeze(0)
        ).item()
        self.t += 1

        reward = cal_reward(self.f[self.t].item(), action.item())

        if reward < 0:
            self.init_data()
            return self.state, -1, True, {}

        if self.t == 50:
            done = True
            self.init_data()
        else:
            done = False
        info = {}
        return self.state, reward, done, info

    def reset(self):
        self.init_obs()
        return self.state

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass

    def init_obs(self):
        self.state[0] = self.f[:8]
        self.state[1] = self.f[:8]
        self.t = 8
        if not config.HAS_LOAD:
            load_checkpoint("checkpoint.pth.tar", self.model, self.optimizer, config.LSTM_LEARNING_RATE)
            config.HAS_LOAD = True
        self.state[2][0] = self.model(
            torch.from_numpy(self.state[0].astype(np.float32)).to(config.DEVICE).unsqueeze(0)
        ).item()

    def init_data(self):
        _, self.f = generator.generate()
        self.f = normalize(self.f)
        self.t = 0


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env

    env = FaaSEnv()
    check_env(env)
