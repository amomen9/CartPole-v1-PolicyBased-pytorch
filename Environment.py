#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""
import matplotlib
# matplotlib.use('Qt5Agg') # 'TkAgg'
matplotlib.use('TkAgg') # 'TkAgg'
import numpy as np
import matplotlib.pyplot as plt
import time
import gymnasium as gym
import functions as fn

# Begin Class CartPoleEnvironment ##############################################################
class CartPoleEnvironment:
    ''' CartPole-v1 environment wrapper for REINFORCE & Actor-Critic training
        (based on gymnasium CartPole-v1, https://gymnasium.farama.org/environments/classic_control/cart_pole/)
        Wraps the gymnasium CartPole environment in a class interface similar to the course framework.
    '''
    
    def __init__(self, max_episode_length=500, render_mode="rgb_array",seed=None):

        self.max_episode_length = max_episode_length
        self.render_mode        = render_mode
        self.n_actions          = 2    # 0: push left, 1: push right
        self.n_observations     = 4    # cart position, cart velocity, pole angle, pole angular velocity

        # Create gymnasium environment
        self.env = gym.make("CartPole-v1", render_mode=render_mode,
                            max_episode_steps=max_episode_length) # max_episode_steps caps each episode at max_episode_length steps.

        # Initialize figures
        self.fig = None

        # Reset the environment
        self.obs,self.info=self.reset(seed=seed)

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


def full_argmax(x):
    ''' Own variant of np.argmax, since np.argmax only returns the first occurence of the max '''
    return np.where(x == np.max(x))[0]            

def test():
    start_time = time.perf_counter()
    env=CartPoleEnvironment(max_episode_length=500, render_mode="rgb_array")
    CartPole_plot=fn.CartPoleAgentPlot(env, title="Test CartPole Agent Plot", curve_plot=False)
    anim, steps = CartPole_plot.test_one_episode(env, policy="test_policy")    
    plt.show()
    
    #print(Q_sa)
    # Test
    # Keep outputs in memory, to flush once at the end
    log_lines = []
    # for t in range(n_test_steps):
    #     #print(s)
    #     a             = np.random.randint(4) # sample random action    
    #     s_next,r,done = env.step(a) # execute action in the environment
    #     p_sas,r_sas   = env.model(s,a)
    #     #print("State {}, Action {}, Reward {}, Next state {}, Done {}, p(s'|s,a) {}, r(s,a,s') {}".format(s,a,r,s_next,done,p_sas,r_sas))
    #     env.render(Q_sa=Q_sa,plot_optimal_policy=True,step_pause=step_pause) # display the environment
    # 
    #     if done: 
    #         s = env.reset()
    #     else: 
    #         s = s_next
    #     log_lines.append(f"{s}\n")
        
    total_execution_time = time.perf_counter() - start_time

    with open("output_Environment.py.log", "w", encoding="utf-8") as f:
        f.writelines(f"Total execution time: {total_execution_time:.3f} seconds" + "\n")
        f.writelines(f"Number of steps: {steps}" + "\n")
    with open("output_Environment.py.log", "a", encoding="utf-8") as f:
        f.writelines(log_lines)
    

               
if __name__ == '__main__':
    test()