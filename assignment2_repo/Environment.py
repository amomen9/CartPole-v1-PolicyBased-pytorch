#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import gymnasium as gym
import time

# Begin Class CartPoleEnvironment ##############################################################
class CartPoleEnvironment:
    ''' CartPole-v1 environment wrapper for DQN training
        (based on gymnasium CartPole-v1, https://gymnasium.farama.org/environments/classic_control/cart_pole/)
        Wraps the gymnasium CartPole environment in a class interface similar to the course framework.
    '''
    
    def __init__(self, max_episode_length=5000, render_mode=None):
        self.max_episode_length = max_episode_length
        self.render_mode        = render_mode
        self.n_actions          = 2    # 0: push left, 1: push right
        self.n_observations     = 4    # cart position, cart velocity, pole angle, pole angular velocity

        # Create gymnasium environment
        self.env = gym.make("CartPole-v1", render_mode=render_mode,
                            max_episode_steps=max_episode_length)

        # Initialize figures
        self.fig = None

        # Reset the environment
        self.reset()

    def reset(self, seed=None):
        ''' Reset the environment to initial state.
        Returns the initial observation and info dict. '''
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, a):
        ''' Forward the environment based on action a.
        Returns next_obs, reward, terminated, truncated, info '''
        return self.env.step(a)

    def render(self):
        ''' Render the environment '''
        return self.env.render()

    def close(self):
        ''' Close the environment '''
        self.env.close()
# End Class CartPoleEnvironment ##############################################################

################[ Environment Setup Block          ]################

######### Variables ###########
max_episode_length = 5000


def make_env(max_episode_length=max_episode_length, render_mode=None):
    return gym.make("CartPole-v1", render_mode=render_mode, max_episode_steps=max_episode_length)


env = make_env(max_episode_length=max_episode_length, render_mode=None)  # truncation condition is max_episode_length steps

# beautify the plots
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('animation', html='jshtml')
####################################################################

################[ Basic Policy Function            ]################
def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1  # go left if leaning left, otherwise go right
####################################################################

################[ Update Scene Function            ]################
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,
####################################################################

################[ Plot Animation Function          ]################
def plot_animation(frames, repeat=False, interval=40):
    print(plt.rcParams['animation.embed_limit'])
    print(f"Number of frames: {len(frames)}")
    minimum_duration_ms = 3000
    interval = max(interval, (minimum_duration_ms + len(frames) - 1) // len(frames))
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = matplotlib.animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    return anim
####################################################################

################[ Show One Episode Function (plot) ]################
def show_one_episode(policy, seed=42, max_episode_length=None):
    truncation_limit = max_episode_length if max_episode_length is not None else globals()["max_episode_length"]
    episode_env = make_env(max_episode_length=truncation_limit, render_mode="rgb_array")

    frames = []
    obs, info = episode_env.reset(seed=seed)
    while True:
        frames.append(episode_env.render())
        action = policy(obs)
        obs, reward, done, truncated, info = episode_env.step(action)
        if done or truncated:
            frames.append(episode_env.render())
            break

    episode_env.close()
    return plot_animation(frames)
####################################################################

def test():
    start_time = time.perf_counter()
    # Hyperparameters
    n_test_steps = 100

    # Initialize environment
    test_env = CartPoleEnvironment(max_episode_length=500)
    obs, info = test_env.reset(seed=42)

    # Test
    # Keep outputs in memory, to flush once at the end
    log_lines = []
    total_reward = 0.0
    for t in range(n_test_steps):
        a = np.random.randint(test_env.n_actions)  # sample random action
        obs, reward, terminated, truncated, info = test_env.step(a)  # execute action in the environment
        total_reward += reward

        if terminated or truncated:
            log_lines.append(f"Episode ended at step {t}, total_reward={total_reward}\n")
            total_reward = 0.0
            obs, info = test_env.reset()

        log_lines.append(f"{obs}\n")

    test_env.close()
    total_execution_time = time.perf_counter() - start_time

    with open("output_Environment.py.log", "w", encoding="utf-8") as f:
        f.writelines(f"Total execution time: {total_execution_time:.3f} seconds" + "\n")
        f.writelines(f"Number of steps: {n_test_steps}" + "\n")
    with open("output_Environment.py.log", "a", encoding="utf-8") as f:
        f.writelines(log_lines)


if __name__ == '__main__':
    test()