from stable_baselines3 import DQN, SAC, PPO, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from FaaSEnv import FaaSEnv

if __name__ == '__main__':
    env = FaaSEnv()
    # env = DummyVecEnv([lambda: env])

    model = PPO(
        "MlpPolicy",
        env=env,
        learning_rate=3e-4,
        batch_size=32,
        gamma=0.99,
        tensorboard_log="./tensorboard/"
    )
    model.learn(total_timesteps=10000)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    env.close()
    print(mean_reward, std_reward)

    obs = env.reset()
    for _ in range(10):
        action, state = model.predict(observation=obs)
        print(action)
        obs, reward, done, info = env.step(action)
        env.render()
