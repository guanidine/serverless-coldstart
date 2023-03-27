from gym.envs.registration import register

if __name__ == '__main__':
    register(
        id='FaaSEnv-v0',
        entry_point='gym.envs.classic_control:FaaSEnv',
        max_episode_steps=1000,
    )
