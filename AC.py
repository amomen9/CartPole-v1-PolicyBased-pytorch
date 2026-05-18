"Import modules"
import os
from typing import cast

import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time

"Set the niceness to allow for other process to also still claim some CPU time"
# import psutil
# psutil.Process().nice(10)

"Import functions and classes"
from Agent import BaseAgent
from Library import average_over_repetitions, LearningCurvePlot, smooth
from Helper import _create_step_progress_bar
import Library as fn
import Environment as environ


class AC_Agent(BaseAgent):

    def __init__(self, actor_hidden_nn=np.array([16, 16]),
                 critic_hidden_nn=np.array([64, 64]),
                 actor_lr=0.001, critic_lr=0.001,
                 gamma=0.99):

        super().__init__(
            actor_hidden_nn=actor_hidden_nn,
            critic_hidden_nn=critic_hidden_nn,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            use_critic=True,
            critic_type='V',
        )

    def update(self, **kwargs):
        """
        Updates the actor and critic networks with
        the given states, rewards and log_probs
        """
        states = kwargs["states"]
        rewards = kwargs["rewards"]
        log_probs = kwargs["log_probs"]

        "Determine the monte carlo return"
        G_t = self.compute_discounted_returns(rewards).detach()

        "We need to stack the list of tensors into one tensor"
        log_probs = torch.stack(log_probs)

        "Determine the policy loss. Since we're summing tensors we need to not use np.sum"
        actor_loss = -torch.sum(G_t * log_probs)

        "Make a tensor of the states"
        states_tensor = torch.as_tensor(states, dtype=torch.float32)

        "Make sure the dimension goes from (something, 1) to (something,)"
        assert self.critic is not None
        assert self.critic_optimizer is not None
        critic_values = self.critic(states_tensor).squeeze()

        "Determine the value loss. Since we're summing tensors we need to not use np.sum"
        critic_loss = torch.sum((G_t - critic_values) ** 2)

        """
        Reset the actor loss gradient to zero
        Determine how weights need to be updated
        and update the weights
        """
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        "Now do the same for the critic part"
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_vectorised(self, states, actions, rewards):
        """

        """

        "Make a tensor of the states and one of the actions"
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)

        "Calculate the log_probs as done in select_action"
        logit = self.actor(states_tensor)
        prob = torch.sigmoid(logit)
        dist = torch.distributions.Bernoulli(probs=prob)
        log_probs = dist.log_prob(actions_tensor.unsqueeze(1)).squeeze()

        "Determine the monte carlo return"
        G_t = self.compute_discounted_returns(rewards).detach()

        "Determine the policy loss. Since we're summing tensors we need to not use np.sum"
        actor_loss = -torch.sum(G_t * log_probs)

        "Make sure the dimension goes from (something, 1) to (something,)"
        assert self.critic is not None
        assert self.critic_optimizer is not None
        critic_values = self.critic(states_tensor).squeeze()

        "Determine the value loss. Since we're summing tensors we need to not use np.sum"
        critic_loss = torch.sum((G_t - critic_values) ** 2)

        """
        Reset the actor loss gradient to zero
        Determine how weights need to be updated
        and update the weights
        """
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        "Now do the same for the critic part"
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


def run_ac(
    agent: AC_Agent,
    env,
    n_timesteps=1000000,
    eval_interval=250,
    enable_progress_bar=True,
    progress_bar_desc="Env Steps",
    progress_bar_position=None,
    shared_step_counter=None,
    max_eval_episode_length=None,
    eval_with_env_episode_trials: bool = True,
    n_eval_episodes: int = 5,
):
    """
    Run one AC training repetition.

    Progress handling matches the REINFORCE shared-counter pattern:
    - optional local tqdm when running single-process
    - optional shared_step_counter for parent-owned parallel progress bars

    Evaluation at each eval_interval:
    - default: use fast proxy `last_episode_return`
    - when enabled: use greedy environment episode trials via `agent.evaluate()`
    """



    "Keep track of the actions, states, log_probs and rewards"
    states = []
    log_probs = []
    rewards = []

    "Retrieve the initial state from the environment"
    state, info = env.reset()

    "Introduce some other useful parameters"
    iteration = 0
    episode_done = False

    "Keep track of return and timestep at evaluated timesteps"
    eval_return = []
    eval_timestep = []

    "Build a progressbar"
    pbar = _create_step_progress_bar(
        total=n_timesteps,
        desc=progress_bar_desc,
        position=progress_bar_position,
        leave=True,
    ) if enable_progress_bar else None

    last_progress_update = 0
    if max_eval_episode_length is None:
        max_eval_episode_length = env.max_episode_length

    try:
        "Calculate episode return for evaluation. This is a noisy estimate of the current policy performance, but it's much faster than calling agent.evaluate() at every eval_interval, which would significantly increase the runtime of your experiments."
        episode_return = 0
        last_episode_return = 0
        "Loop until the max. number of timesteps is reached, but don't cut off an episode early"
        while iteration < n_timesteps:
            "Choose the action to take"
            action, log_prob = agent.select_action(state)
            
            
            "Receive the next state, reward and whether the agent is terminated/truncated"
            next_state, reward, terminated, truncated, info = env.step(action)

            "The episode is done if the agent terminates/truncates"
            episode_done = terminated or truncated

            "Store the reward, state and log probability"
            rewards.append(reward)
            states.append(state)
            log_probs.append(log_prob)
            episode_return += reward

            "Update the state"
            state = next_state

            "Increase the iteration counter by 1"
            iteration += 1

            "Update the progress bar and shared counter periodically"
            if (iteration - last_progress_update) >= 512 or iteration >= n_timesteps:
                progress_delta = iteration - last_progress_update
                if pbar is not None:
                    pbar.update(progress_delta)
                if shared_step_counter is not None:
                    shared_step_counter.value = min(iteration, n_timesteps)
                last_progress_update = iteration

            "Store the timestep and return at the correct interval"
            if iteration % eval_interval == 0:
                if eval_with_env_episode_trials:
                    eval_return.append(
                        agent.evaluate(
                            n_eval_episodes=n_eval_episodes,
                            max_steps=max_eval_episode_length,
                        )
                    )
                else:
                    # Fast proxy: return from the last completed episode
                    eval_return.append(last_episode_return)
                eval_timestep.append(iteration)

            "If the agent terminates:"
            if episode_done:
                "Reset the environment"
                state, info = env.reset()
                last_episode_return = episode_return
                episode_return = 0

                "Update the actor and critic networks on the completed episode"
                agent.update(states=np.array(states), rewards=np.array(rewards), log_probs=log_probs)

                "Reset the rewards and log probs array"
                rewards = []
                log_probs = []
                states = []

    finally:
        if shared_step_counter is not None:
            shared_step_counter.value = min(iteration, n_timesteps)
        if pbar is not None:
            pbar.close()
        env.close()

    "When we're done we want the timesteps and the return"
    return eval_return, eval_timestep


class CartPoleEnvironments:
    ''' CartPole-v1 environment wrapper for REINFORCE & Actor-Critic training
        (based on gymnasium CartPole-v1, https://gymnasium.farama.org/environments/classic_control/cart_pole/)
        Wraps the gymnasium CartPole environment in a class interface similar to the course framework.
    '''

    def __init__(self, max_episode_length=500, render_mode="rgb_array", seed=None, n_envs=3):

        self.max_episode_length = max_episode_length
        self.render_mode = render_mode
        self.n_actions = 2    # 0: push left, 1: push right
        self.n_observations = 4    # cart position, cart velocity, pole angle, pole angular velocity
        self.n_envs = n_envs

        # Create gymnasium environment
        self.envs = gym.make_vec(
            "CartPole-v1",
            num_envs=n_envs,
            vectorization_mode="sync",
            render_mode=render_mode,
            max_episode_steps=max_episode_length,
        )  # max_episode_steps caps each episode at max_episode_length steps.
        print(self.envs.num_envs, self.envs.action_space)
        # Initialize figures
        self.fig = None

        # Reset the environment
        self.obs, self.info = self.reset(seed=seed)

    def reset(self, seed=None):
        ''' Reset the environment to initial state.
        Returns the initial observation and info dict. '''
        obs, info = self.envs.reset(seed=seed)
        return obs, info

    def step(self, a):
        ''' Forward the environment based on action a.
        Returns next_obs, reward, terminated, truncated, info '''
        return self.envs.step(a)

    def render(self):
        ''' Render the environment '''
        return self.envs.render()

    def close(self):
        ''' Close the environment '''
        self.envs.close()


def run_ac_vectorised(agent: AC_Agent, envs, n_timesteps=1000000, eval_interval=250):
    """

    """
    "Read the number of environments in the vector"
    n_envs = envs.n_envs
    eval_env = environ.CartPoleEnvironment(
        max_episode_length=envs.max_episode_length,
        render_mode=envs.render_mode,
    )

    "Keep track of the actions, states, log_probs and rewards. This needs to be done per environment"
    actions = [[] for _ in range(n_envs)]
    states = [[] for _ in range(n_envs)]
    log_probs = [[] for _ in range(n_envs)]
    rewards = [[] for _ in range(n_envs)]

    "Retrieve the initial state from the environment"
    state_vec, info = envs.reset()

    "Introduce some useful parameters"
    last_eval = 0
    iteration = 0
    done_flags: np.ndarray = np.array([False] * n_envs, dtype=bool)

    "Keep track of return and timestep at evaluated timesteps"
    eval_return = []
    eval_timestep = []

    "Build a progressbar"
    progressbar = _create_step_progress_bar(
        total=n_timesteps,
        desc="Running the AC algorithm",
        leave=True,
    )

    "Loop until the max. number of timesteps is reached, but don't cut off an episode early"
    while iteration < n_timesteps:
        # for i in range(n_timesteps):
        "Choose the action to take"
        state_tensor = torch.as_tensor(state_vec, dtype=torch.float32)
        logit = agent.actor(state_tensor)
        prob = torch.sigmoid(logit)
        dist = torch.distributions.Bernoulli(probs=prob)
        action_tensor = dist.sample()
        log_prob_tensor = dist.log_prob(action_tensor)
        action_vec = np.asarray(action_tensor.squeeze(-1).cpu().numpy(), dtype=np.int32)
        log_prob_vec = np.asarray(log_prob_tensor.squeeze(-1).cpu().numpy(), dtype=np.float32)

        "Receive the next state, reward and whether the agent is terminated/truncated"
        next_state_vec, reward_vec, terminated_vec, truncated_vec, info = envs.step(action_vec)

        "The episode is done if the agent terminates/truncates"
        done_flags = np.asarray(terminated_vec, dtype=bool) | np.asarray(truncated_vec, dtype=bool)

        "Store the reward, state and log probability. Do this for all environments"
        for i in range(n_envs):
            states[i].append(state_vec[i])
            actions[i].append(int(action_vec[i]))
            rewards[i].append(float(reward_vec[i]))
            log_probs[i].append(float(log_prob_vec[i]))

            "Check if the environment is done, if so run the update method"
            if bool(done_flags[i]):
                agent.update_vectorised(np.array(states[i]), np.array(actions[i]), np.array(rewards[i]))

                "Reset the environment entries"
                states[i], actions[i], rewards[i], log_probs[i] = [], [], [], []

        "Update the state"
        state_vec = next_state_vec

        "Increase the iteration counter by 1"
        iteration += n_envs

        "Update the progressbar"
        progressbar.update(n_envs)

        if iteration - last_eval >= eval_interval:
            "Evaluate the agent's performance"
            eval_return.append(
                agent.evaluate(
                    n_eval_episodes=5,
                    max_steps=eval_env.max_episode_length,
                )
            )

            "Store the current timestep"
            last_eval = iteration
            eval_timestep.append(iteration)

    "When we're done we want the timesteps and the return"
    return eval_timestep, eval_return


def run_parallel(workers=5, n_timesteps=1000000, actor_hidden_nn=[16, 16], critic_hidden_nn=[64, 64], actor_rl=0.001, critic_rl=0.001, gamma=0.99, path="run.npy"):
    """
    Uses the shared parallel runner from Helper so AC matches REINFORCE's
    progress-reporting behavior.
    """
    os.makedirs("Data", exist_ok=True)
    start_time = time.time()

    curve_result = average_over_repetitions(
        method="ac",
        n_repetitions=workers,
        n_timesteps=n_timesteps,
        eval_interval=2500,
        max_episode_length=500,
        actor_lr=actor_rl,
        gamma=gamma,
        actor_hidden_nn=np.asarray(actor_hidden_nn, dtype=np.int32),
        critic_hidden_nn=np.asarray(critic_hidden_nn, dtype=np.int32),
        critic_lr=critic_rl,
        return_raw=True,
    )
    learning_curve, learning_curve_std, timestep, raw_returns = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        curve_result,
    )

    np.save("Data/" + path, np.array([timestep, raw_returns], dtype=object))

    with open("Data/timing_log.txt", "a") as f:
        f.write(f"Creating {path} took {(time.time() - start_time):.5g} seconds\n")

    # "Calculate the mean and std"
    # mean = np.mean(returns,axis=0)
    # std = np.std(returns,axis=0)

    # plt.figure()

    # "Add the mean plot to the figure"
    # plt.plot(timestep,mean,color='blue')

    # "Add the error region to the figure"
    # plt.fill_between(timestep, mean, mean + std,alpha=0.1,color='blue')
    # plt.fill_between(timestep, mean, mean - std,alpha=0.1,color='blue')
    # plt.show()


def load_data_test():

    "Load the timestep and returns"
    timestep, returns = np.load("Data/AC_test.npy", allow_pickle=True)

    print(timestep.shape, returns.shape)

    "Calculate the mean and std"
    mean = np.mean(returns, axis=0)
    std = np.std(returns, axis=0)

    plt.figure()

    "Add the mean plot to the figure"
    plt.plot(timestep, mean, color='blue')

    "Add the error region to the figure"
    plt.fill_between(timestep, mean, mean + std, alpha=0.1, color='blue')
    plt.fill_between(timestep, mean, mean - std, alpha=0.1, color='blue')

    # plt.show()

    "Create a figure"
    plot = fn.LearningCurvePlot("Title test")
    plot.add_curve(timestep, mean, "AC")
    plot.add_shaded_ci(timestep, mean, std, n=0.1, alpha=0.9, fill_opacity=0.4)
    plt.show()


"Import modules"
import numpy as np
import os

"Import relevant functions"
workers          = 5
actor_hidden_nn  = [16,16]
critic_hidden_nn = [64,64]
actor_lr         = 0.001
critic_lr        = 0.001
gamma            = 0.99
n_timesteps      = 1000000
colours          = ["red","blue","purple","green"]
linestyles       = ["solid","--",":","-."]

"Set the window size for plotting"
window_size      = 42

"Define the parameters that we want to explore"
actor_lrs  = [0.001, 0.005, 0.01, 0.05]
critic_lrs = [0.001, 0.005, 0.01, 0.05]
actor_nns  = [[16,16],[32,32],[64,64]]
critic_nns = [[32,32],[64,64],[128,128]]


def run_actor_lr_experiments():
    """
    Check the influence of the learning rate
    of the actor by using different learning
    rates and keeping other parameters constant
    """

    "Run the experiments for the different learning rates"
    for actor_lr_i in actor_lrs:
        "Define the path to the file we want to store"
        path = f"HP_actorlr_{actor_lr_i}.npy"

        "Try to run the experiment"
        try:
            run_parallel(workers=workers,n_timesteps=n_timesteps,
                     actor_hidden_nn=actor_hidden_nn, actor_rl=actor_lr_i,
                     critic_hidden_nn=critic_hidden_nn,critic_rl=critic_lr,
                     gamma=gamma,path=path)
        except:
            "If this fails, report it"
            with open("Data/timing_log.txt","a") as f:
                f.write(f"Something went wrong creating {path}!\n")

    "Try to plot the data"
    try:
        "Make a figure"
        plot = LearningCurvePlot("Actor learning rates")

        for i in range(len(actor_lrs)):
            "Load the data"
            timestep, returns = np.load("Data/"+f"HP_actorlr_{actor_lrs[i]}.npy",allow_pickle=True)

            "Calculate the mean and std"
            mean = np.mean(returns,axis=0)
            std  = np.std(returns,axis=0)

            "Add the relevant curve"
            plot.add_curve(timestep,mean,label=f"LR = {actor_lrs[i]}",color=colours[i],ls=linestyles[i])

            "Add the error region as well"
            plot.add_shaded_ci(timestep,mean,std,n=workers,alpha=0.32,fill_opacity=0.4, y_lower_cap=0, y_upper_cap=500)
        
        "Store the figure"
        plot.save("plots/actor_lr.pdf")
    except:
        "If this went wrong, report it"
        with open("Data/timing_log.txt","a") as f:
            f.write(f"Something went wrong creating actor_lr.pdf\n")
        
def run_critic_lr_experiments():
    """
    Check the influence of the learning rate
    of the critic by using different learning
    rates and keeping other parameters constant
    """

    "Run the experiments for the different learning rates"
    for critic_lrs_i in critic_lrs:
        "Define the path to the file we want to store"
        path = f"HP_criticlr_{critic_lrs_i}.npy"

        "Try to run the experiment"
        try:        
            run_parallel(workers=workers,n_timesteps=n_timesteps,
                        actor_hidden_nn=actor_hidden_nn, actor_rl=actor_lr,
                        critic_hidden_nn=critic_hidden_nn,critic_rl=critic_lrs_i,
                        gamma=gamma,path=path)
        except:
            "If this fails, report it"
            with open("Data/timing_log.txt","a") as f:
                f.write(f"Something went wrong creating {path}!\n")
    
    "Try to plot the data"
    try:
        "Make a figure"
        plot = LearningCurvePlot("Critic learning rates")

        "Plot the data"
        for i in range(len(actor_lrs)):
            "Load the data"
            timestep, returns = np.load("Data/"+f"HP_criticlr_{critic_lrs[i]}.npy",allow_pickle=True)

            "Calculate the mean and std"
            mean = np.mean(returns,axis=0)
            std  = np.std(returns,axis=0)

            "Add the relevant curve"
            plot.add_curve(timestep,mean,label=f"LR = {critic_lrs[i]}",color=colours[i],ls=linestyles[i])

            "Add the error region as well"
            plot.add_shaded_ci(timestep,mean,std,n=workers,alpha=0.32,fill_opacity=0.4, y_lower_cap=0, y_upper_cap=500)
        
        "Store the figure"
        plot.save("plots/critic_lr.pdf")
    except:
        "If this fails, report it"
        with open("Data/timing_log.txt","a") as f:
            f.write(f"Something went wrong creating critic_lr.pdf\n")

def run_actor_nn_experiments():
    """
    Check the influence of the architecture of the actor
    hidden layers by using different architectures and 
    keeping other parameters constant
    """

    "Run the experiments for the different architectures"
    for actor_nns_i in actor_nns:
        "Define the path to the file we want to store"
        path = f"HP_actornn_{actor_nns_i}.npy"

        "Try to run the experiment"
        try:        
            run_parallel(workers=workers,n_timesteps=n_timesteps,
                        actor_hidden_nn=actor_nns_i, actor_rl=actor_lr,
                        critic_hidden_nn=critic_hidden_nn,critic_rl=critic_lr,
                        gamma=gamma,path=path)
        except:
            "If this fails, report it"
            with open("Data/timing_log.txt","a") as f:
                f.write(f"Something went wrong creating {path}!\n")
    
    "Try to plot the data"
    try:
        "Make a figure"
        plot = LearningCurvePlot("Actor hidden layer architecture")

        "Plot the data"
        for i in range(len(actor_nns)):
            "Load the data"
            timestep, returns = np.load("Data/"+f"HP_actornn_{actor_nns[i]}.npy",allow_pickle=True)

            "Calculate the mean and std"
            mean = np.mean(returns,axis=0)
            std  = np.std(returns,axis=0)

            "Add the relevant curve"
            plot.add_curve(timestep,mean,label=f"Hidden layer = {actor_nns[i]}",color=colours[i],ls=linestyles[i])

            "Add the error region as well"
            plot.add_shaded_ci(timestep,mean,std,n=workers,alpha=0.32,fill_opacity=0.4, y_lower_cap=0, y_upper_cap=500)
        
        "Store the figure"
        plot.save("plots/actor_nn.pdf")
    except:
        "If this fails, report it"
        with open("Data/timing_log.txt","a") as f:
            f.write(f"Something went wrong creating actor_nn.pdf\n")
        
def run_critic_nn_experiments(plot_only=False):
    """
    Check the influence of the architecture of the critic
    hidden layers by using different architectures and 
    keeping other parameters constant
    """

    "If we only want to plot, we can do that!"
    if not plot_only:
        "Run the experiments for the different architectures"
        for critic_nns_i in critic_nns:
            "Define the path to the file we want to store"
            path = f"HP_criticnn_{critic_nns_i}.npy"

            "Try to run the experiment"
            try:        
                run_parallel(workers=workers,n_timesteps=n_timesteps,
                            actor_hidden_nn=actor_hidden_nn, actor_rl=actor_lr,
                            critic_hidden_nn=critic_nns_i,critic_rl=critic_lr,
                            gamma=gamma,path=path)
            except:
                "If this fails, report it"
                with open("Data/timing_log.txt","a") as f:
                    f.write(f"Something went wrong creating {path}!\n")
    
    "Try to plot the data"
    try:
        "Make a figure"
        plot = LearningCurvePlot("Critic hidden layer architecture")

        "Plot the data"
        for i in range(len(critic_nns)):
            "Load the data"
            timestep, returns = np.load("Data/"+f"HP_criticnn_{critic_nns[i]}.npy",allow_pickle=True)

            "Calculate the mean and std"
            mean = np.mean(returns,axis=0)
            std  = np.std(returns,axis=0)

            "Add the relevant curve"
            plot.add_curve(timestep,mean,label=f"Hidden layer = {critic_nns[i]}",color=colours[i],ls=linestyles[i])

            "Add the error region as well"
            plot.add_shaded_ci(timestep,mean,std,n=workers,alpha=0.32,fill_opacity=0.4, y_lower_cap=0, y_upper_cap=500)
        
        "Store the figure"
        plot.save("plots/critic_nn.pdf")
    except:
        "If this fails, report it"
        with open("Data/timing_log.txt","a") as f:
            f.write(f"Something went wrong creating critic_nn.pdf\n")

def plot_HP_tests():
    """
    Plots the hyperparameter tests
    """

    "Create the plot for the actor learning rate testing"
    plot1 = LearningCurvePlot("Actor learning rate")
    
    "Loop over all the learning rates"
    for i in range(len(actor_lrs)):
        "Load the data"
        timestep1, returns1 = np.load("Data/"+f"HP_actorlr_{actor_lrs[i]}.npy",allow_pickle=True)

        "Calculate the mean and std"
        mean1 = np.mean(returns1,axis=0)
        std1  = np.std(returns1,axis=0)

        "Smooth these"
        mean1_sm = smooth(mean1, window = window_size)
        std1_sm  = smooth(std1, window = window_size)

        "Add the smoothed curves to the plot"
        plot1.add_curve(timestep1,mean1_sm,label=f"LR = {actor_lrs[i]}",color=colours[i],ls=linestyles[i])
        plot1.add_shaded_ci(timestep1,mean1_sm,std1_sm,n=workers,alpha=0.32,fill_opacity=0.4, y_lower_cap=0, y_upper_cap=500)
    
    "Save this figure"
    plot1.save("plots/HP_actor_lr.pdf")
    

    "Create the plot for the critic learning rate testing"
    plot2 = LearningCurvePlot("Critic learning rate")

    "Loop over all learning rates"
    for i in range(len(critic_lrs)):
        "Load the data"
        timestep2, returns2 = np.load("Data/"+f"HP_criticlr_{critic_lrs[i]}.npy",allow_pickle=True)

        "Calculate the mean and std"
        mean2 = np.mean(returns2,axis=0)
        std2  = np.std(returns2,axis=0)

        "Smooth these"
        mean2_sm = smooth(mean2, window = window_size)
        std2_sm = smooth(std2, window = window_size)

        "Add the smoothed curves to the plot"
        plot2.add_curve(timestep2,mean2_sm,label=f"LR = {critic_lrs[i]}",color=colours[i],ls=linestyles[i])
        plot2.add_shaded_ci(timestep2,mean2_sm,std2_sm,n=workers,alpha=0.32,fill_opacity=0.4, y_lower_cap=0, y_upper_cap=500)

    plot2.save("plots/HP_critic_lr.pdf")

    "Create the plot for the actor NN architecture testing"
    plot3 = LearningCurvePlot("Actor hidden layer architecture")

    "Loop over all actor architectures"
    for i in range(len(actor_nns)):
        "Load the data"
        timestep3, returns3 = np.load("Data/"+f"HP_actornn_{actor_nns[i]}.npy",allow_pickle=True)

        "Calculate the mean and std"
        mean3 = np.mean(returns3,axis=0)
        std3  = np.std(returns3,axis=0)

        "Smooth these"
        mean3_sm = smooth(mean3, window = window_size)
        std3_sm = smooth(std3, window = window_size)

        "Add the smoothed curves to the plot"
        plot3.add_curve(timestep3,mean3_sm,label=f"Hidden layer = {actor_nns[i]}",color=colours[i],ls=linestyles[i])
        plot3.add_shaded_ci(timestep3,mean3_sm,std3_sm,n=workers,alpha=0.32,fill_opacity=0.4, y_lower_cap=0, y_upper_cap=500)

    plot3.save("plots/HP_actor_nn.pdf")
    
    "Create the plot for the critic NN architecture testing"
    plot4 = LearningCurvePlot("Critic hidden layer architecture")

    "Loop over all actor architectures"
    for i in range(len(critic_nns)):
        "Load the data"
        timestep4, returns4 = np.load("Data/"+f"HP_criticnn_{critic_nns[i]}.npy",allow_pickle=True)

        "Calculate the mean and std"
        mean4 = np.mean(returns4,axis=0)
        std4  = np.std(returns4,axis=0)

        "Smooth these"
        mean4_sm = smooth(mean4, window = window_size)
        std4_sm  = smooth(std4, window = window_size)

        "Add the smoothed curves to the plot"
        plot4.add_curve(timestep4,mean4_sm,label=f"Hidden layer = {critic_nns[i]}",color=colours[i],ls=linestyles[i])
        plot4.add_shaded_ci(timestep4,mean4_sm,std4_sm,n=workers,alpha=0.32,fill_opacity=0.4, y_lower_cap=0, y_upper_cap=500)
    
    plot4.save("plots/HP_critic_nn.pdf")

def final_run(plotting=False):
    """
    Runs the AC algorithm with the optimised
    hyperparameters
    """

    "Set the chosen parameters here"
    a_lr_best = 0.001
    c_lr_best = 0.001
    a_nn_best = [64,64]
    c_nn_best = [128,128]

    "Run the algorithm with the provided hyperparameters"
    run_parallel(workers=workers,n_timesteps=n_timesteps,
                     actor_hidden_nn=a_nn_best, actor_rl=a_lr_best,
                     critic_hidden_nn=c_nn_best,critic_rl=c_lr_best,
                     gamma=gamma,path="AC_final_run.npy")
    
    if plotting:
        "Load the data"
        timestep, returns = np.load("Data/AC_final_run.npy",allow_pickle=True)

        "Calculate the mean and std"
        mean = np.mean(returns,axis=0)
        std  = np.std(returns,axis=0)

        "Smooth these"
        mean_sm = smooth(mean, window = window_size)
        std_sm  = smooth(std, window = window_size)

        "Create a figure"
        plot = LearningCurvePlot("AC run with best parameters")

        "Add the smoothed curves to the plot"
        plot.add_curve(timestep,mean_sm,label="AC best param",color="black",ls="solid")
        plot.add_shaded_ci(timestep,mean_sm,std_sm,n=workers,alpha=0.32,fill_opacity=0.4, y_lower_cap=0, y_upper_cap=500)

        plot.save("AC_best_run.pdf")
