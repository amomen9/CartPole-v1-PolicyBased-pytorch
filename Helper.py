#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as visplt
import os
from scipy.signal import savgol_filter
# Import Environment lazily inside __main__ to avoid circular imports during
# multiprocessing worker startup.
# from statsmodels.nonparametric.kernel_regression import KernelReg


def _create_step_progress_bar(total, desc, position=None, leave=True):
    """Create a tqdm progress bar with the shared project formatting."""
    from tqdm import tqdm

    tqdm_kwargs = {
        "total": total,
        "desc": desc,
        "unit": "step",
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        "dynamic_ncols": True,
        "leave": leave,
    }
    if position is not None:
        tqdm_kwargs["position"] = int(position)
    return tqdm(**tqdm_kwargs)

# Begin Class LearningCurvePlot ##############################################################
class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Timestep')
        self.ax.set_ylabel('Episode Return')      
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self, x, y, label=None, ls="solid", color=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        plot_kwargs = {"ls": ls}
        if color is not None:
            plot_kwargs["color"] = color
        if label is not None:
            self.ax.plot(x, y, label=label, **plot_kwargs)
        else:
            self.ax.plot(x, y, **plot_kwargs)

    def add_shaded_ci(self, x, y_mean, y_std, n, alpha=0.2, fill_opacity=0.15, y_lower_cap=None, y_upper_cap=None, color=None):
        '''Add a shaded confidence band around the mean curve.
        alpha controls CI significance (e.g., 0.05 for 95% CI),
        fill_opacity controls the visual transparency of the shaded area.'''
        from scipy.stats import t as t_dist
        t_crit = t_dist.ppf(1 - alpha / 2, df=max(n - 1, 1))
        margin = t_crit * y_std / np.sqrt(max(n, 1))
        y_lower = y_mean - margin
        y_upper = y_mean + margin
        if y_lower_cap is not None:
            y_lower = np.maximum(y_lower, y_lower_cap)
        if y_upper_cap is not None:
            y_upper = np.minimum(y_upper, y_upper_cap)
        if color is None:
            color = self.ax.get_lines()[-1].get_color()  # match the last plotted line
        self.ax.fill_between(x, y_lower, y_upper,
                             alpha=fill_opacity, color=color)
    
    def set_ylim(self,lower,upper):
        self.ax.set_ylim([lower,upper])

    def add_hline(self,height,label):
        self.ax.axhline(height,ls='--',c='k',label=label)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend(
            fontsize=8,
            handlelength=1.2,
            handletextpad=0.4,
            borderpad=0.25,
            labelspacing=0.25,
            borderaxespad=0.3,
        )
        self.fig.tight_layout()
        output_path = name
        if not os.path.isabs(name):
            os.makedirs("plots", exist_ok=True)
            output_path = os.path.join("plots", os.path.basename(name))
        self.fig.savefig(output_path,dpi=300)
# End Class LearningCurvePlot ##############################################################



def smooth(y, window, poly=2):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

### One suggested simplest policy {eps, 1-eps} is below, however, I implement the one that was mentioned in the assignment instead.
#def egreedy(Qa_s, eps):
#    ''' Qa_s: vector of action values for state s
#        epsilon: exploration parameter '''
#    if np.random.rand() < eps:
#        return np.random.randint(0,len(Qa_s)) # Explore action space
#    else:
#        return argmax(Qa_s) # Exploit learned values

def egreedy(Qa_s, eps):
    """
    Sample one action using epsilon-greedy policy
    Qa_s: 1D array of Q-values for current state's actions
    eps: epsilon in the closed boundary [0,1]
    """
    n_A = len(Qa_s)     # number of actions
    greedy_a = argmax(Qa_s)  # tie breaking argmax()
    # Base probability for all actions, fill probs matrix with the same values (will not sum up to 1 yet)
    probs = np.full(n_A, eps / n_A, dtype=float)
    # Greedy action gets the remaining probability mass (1 - eps) plus its share of the exploration probability (eps/n_A)
    probs[greedy_a] = 1.0 - eps * (n_A - 1) / n_A
    selected_action = np.random.choice(n_A, p=probs)
    # Sample action from this distribution
    return selected_action

def softmax(x, temp):   # aka Boltzmann policy (Mentioned as Boltzmann in the assignment)
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax
    selected_action = np.exp(z)/np.sum(np.exp(z)) # compute softmax
    return selected_action

def argmax(x):
    ''' Own variant of np.argmax with random tie breaking '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)

def linear_anneal(t,T,start,final,percentage):
    ''' Linear annealing scheduler
    t: current timestep
    T: total timesteps
    start: initial value
    final: value after percentage*T steps
    percentage: percentage of T after which annealing finishes
    ''' 
    final_from_T = int(percentage*T)
    if t > final_from_T:
        return final
    else:
        return final + (start - final) * (final_from_T - t)/final_from_T

################[ Policy_NN Class              ]################
def softmax(x, temp):   # aka Boltzmann policy (Mentioned in Assignment 1 as Boltzmann in the assignment)
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax
    probs = np.exp(z)/np.sum(np.exp(z)) # compute softmax
    selected_action = np.random.choice(len(x), p=probs) # Sample action from
    return int(selected_action)
####################################################################



if __name__ == '__main__':
    # Test Learning curve plot
    # x = np.arange(100)
    # y = 0.01*x + np.random.rand(100) - 0.4 # generate some learning curve y
    # LCTest = LearningCurvePlot(title="Test Learning Curve")
    # LCTest.add_curve(x,y,label='method 1')
    # LCTest.add_curve(x,smooth(y,window=35),label='method 1 smoothed')
    # LCTest.save(name='learning_curve_test.png')
    import Environment
    env = Environment.CartPoleEnvironment(max_episode_length=500, render_mode="rgb_array")
    CartPole_plot = CartPoleAgentPlot(env, title="Test CartPole Agent Plot", plot=False)
    anim = CartPole_plot.test_episode(env, policy="test_policy")
    visplt.show()
