from gym.envs.registration import register

if __name__ == '__main__':
    register(
        id='FaaSEnv-v0',
        entry_point='my_module:FaaSEnv',
        kwargs={'arg1': 1, 'arg2': 2}
    )
