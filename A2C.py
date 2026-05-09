# pyright: reportMissingImports=false
from Agent import BaseAgent

import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Any, cast

from Environment import CartPoleEnvironment
from functions import average_over_repetitions, LearningCurvePlot, smooth
from tqdm import tqdm


class A2C_Agent(BaseAgent):
    def __init__(
        self,
        n_agent_state_elements,
        n_actions,
        actor_lr,
        critic_lr,
        gamma,
        TN_step,
        actor_hidden_nn,
        critic_hidden_nn,
    ):
        """Initialise the A2C agent."""

        self.n_agent_state_elements = n_agent_state_elements
        self.n_actions = n_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.TN_step = TN_step

        self.policy = PolicyNetwork(
            n_agent_state_elements,
            n_actions,
            actor_hidden_nn,
            activation=nn.ReLU,
        )
        self.value_func = ValueNetwork(
            n_agent_state_elements,
            critic_hidden_nn,
            activation=nn.ReLU,
        )

        self.opt_actor = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.opt_critic = torch.optim.Adam(self.value_func.parameters(), lr=critic_lr)

    def select_action(self, obs):
        """Sample an action from the policy."""

        probs = self.policy(obs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    def compute_returns(self, rewards, next_states, dones):
        """Compute the n-step return estimate for each timestep."""

        episode_length = len(rewards)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)

        with torch.no_grad():
            v_s_next = self.value_func(next_states_t)

        q_hat = torch.zeros(episode_length)

        for t in range(episode_length):
            ret = 0.0

            for k in range(self.TN_step):
                if t + k >= episode_length:
                    break

                ret += (self.gamma**k) * rewards[t + k]

                if dones[t + k]:
                    break
            else:
                boot_idx = min(t + self.TN_step, episode_length - 1)
                ret += (self.gamma**self.TN_step) * v_s_next[boot_idx].item()

            q_hat[t] = ret

        return q_hat

    def update(self, **kwargs):
        """Update policy and value networks from one trajectory."""

        states = kwargs["states"]
        actions = kwargs["actions"]
        rewards = kwargs["rewards"]
        next_states = kwargs["next_states"]
        dones = kwargs["dones"]

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.long)

        q_hat = self.compute_returns(rewards, next_states, dones)

        v_s = self.value_func(states_t)
        advantages = q_hat - v_s

        critic_loss = (advantages**2).sum()
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        probs = self.policy(states_t)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_t)
        actor_loss = -(advantages.detach() * log_probs).sum()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

    def evaluate(self, n_eval_episodes: int = 30, max_steps: int = 500) -> np.floating:
        """Evaluate the current policy greedily (deterministic).

        Uses environment episode rollouts (slower than the training-loop proxy
        `last_episode_return`).
        """
        env = CartPoleEnvironment(max_episode_length=max_steps, render_mode="rgb_array")

        total_returns: list[float] = []
        for _ in range(n_eval_episodes):
            obs, _ = env.reset()
            ep_return = 0.0

            for _ in range(max_steps):
                with torch.no_grad():
                    probs = self.policy(obs)
                    action = int(torch.argmax(probs, dim=-1).item())

                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)

                if terminated or truncated:
                    break

            total_returns.append(ep_return)

        env.close()
        return np.mean(total_returns)


class PolicyNetwork(nn.Module):
    def __init__(self, n_agent_state_elements, n_actions, hidden_layers, activation):
        super().__init__()

        layers = []
        input_size = n_agent_state_elements

        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(activation())
            input_size = layer_size

        layers.append(nn.Linear(input_size, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return torch.softmax(self.net(x), dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, n_agent_state_elements, hidden_layers, activation):
        super().__init__()

        layers = []
        input_size = n_agent_state_elements

        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(activation())
            input_size = layer_size

        layers.append(nn.Linear(input_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return self.net(x).squeeze(-1)


def run_a2c(
    agent,
    env,
    n_timesteps=1000000,
    eval_interval=250,
    truncation_step=500,
    enable_progress_bar=True,
    progress_bar_desc="Env Steps",
    progress_bar_position=None,
    shared_step_counter=None,
    max_eval_episode_length=None,
    eval_with_env_episode_trials: bool = False,
    n_eval_episodes: int = 5,
):
    """A2C training loop with eval-interval return recording.

    Progress handling mirrors the REINFORCE loop so parent-owned parallel
    tqdm bars behave consistently across algorithms.
    """

    data_count = math.ceil(n_timesteps / eval_interval)
    eval_returns = np.empty(data_count, dtype=np.float32)
    eval_timesteps = np.empty(data_count, dtype=np.int32)
    eval_write_idx = 0

    global_step = 0
    last_episode_return = 0.0

    if max_eval_episode_length is None:
        max_eval_episode_length = env.max_episode_length

    pbar = None
    if enable_progress_bar:
        tqdm_kwargs = {
            "total": n_timesteps,
            "desc": progress_bar_desc,
            "unit": "step",
            "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            "dynamic_ncols": True,
            "leave": True,
        }
        if progress_bar_position is not None:
            tqdm_kwargs["position"] = int(progress_bar_position)
        pbar = tqdm(**tqdm_kwargs)

    last_progress_update = 0

    try:
        while global_step < n_timesteps:
            states, actions, rewards, next_states, dones = [], [], [], [], []
            state, _ = env.reset()
            episode_done = False

            for _ in range(truncation_step):
                action, _ = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(episode_done)

                state = next_state
                global_step += 1

                if (global_step - last_progress_update) >= 512 or global_step >= n_timesteps:
                    progress_delta = global_step - last_progress_update
                    if pbar is not None:
                        pbar.update(progress_delta)
                    if shared_step_counter is not None:
                        shared_step_counter.value = min(global_step, n_timesteps)
                    last_progress_update = global_step

                if global_step % eval_interval == 0:
                    if eval_with_env_episode_trials:
                        eval_returns[eval_write_idx] = agent.evaluate(
                            n_eval_episodes=n_eval_episodes,
                            max_steps=max_eval_episode_length,
                        )
                    else:
                        eval_returns[eval_write_idx] = last_episode_return
                    eval_timesteps[eval_write_idx] = global_step
                    eval_write_idx += 1

                if episode_done or global_step >= n_timesteps:
                    break

            if states:
                agent.update(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    dones=dones,
                )
                last_episode_return = sum(rewards)
                if pbar is not None:
                    pbar.set_postfix_str(f"episode_reward={last_episode_return:.2f}", refresh=False)
    finally:
        if shared_step_counter is not None:
            shared_step_counter.value = min(global_step, n_timesteps)
        if pbar is not None:
            pbar.close()

    if global_step % eval_interval != 0:
        eval_returns[eval_write_idx] = last_episode_return
        eval_timesteps[eval_write_idx] = global_step
        eval_write_idx += 1

    return eval_returns[:eval_write_idx], eval_timesteps[:eval_write_idx]




def best_A2C(n_repetitions, n_timesteps, gamma, smoothing_window):
    """Run A2C with the selected best hyperparameters."""

    best_lr_policy = 0.0001
    best_lr_value_func = 0.01
    best_n_step = 10
    best_hl_policy = np.array([64, 64])
    best_hl_value_func = np.array([128, 128])

    baseline = pd.read_csv("BaselineDataCartPole.csv")

    plot = LearningCurvePlot(title="A2C test")

    a2c_results = average_over_repetitions(
        method="a2c",
        n_repetitions=n_repetitions,
        n_timesteps=n_timesteps,
        eval_interval=250,
        max_episode_length=500,
        actor_lr=best_lr_policy,
        critic_lr=best_lr_value_func,
        gamma=gamma,
        actor_hidden_nn=best_hl_policy,
        critic_hidden_nn=best_hl_value_func,
    )
    mean_returns, std_returns, timesteps = cast(tuple[np.ndarray, np.ndarray, np.ndarray], a2c_results)

    data = np.column_stack([timesteps, mean_returns, std_returns])
    np.save("best_a2c_data.npy", data)

    plot.add_curve(timesteps[::250], smooth(mean_returns[::250], smoothing_window), label="A2C test")
    plot.add_shaded_ci(
        timesteps[::250],
        smooth(mean_returns[::250], smoothing_window),
        smooth(std_returns[::250], smoothing_window),
        n_repetitions,
    )

    plot.add_curve(
        baseline["env_step"].values[3806:],
        baseline["Episode_Return_smooth"].values[3806:],
        label="Baseline",
    )
    plot.add_hline(500, label="Optimal")
    plot.save("best_a2c.pdf")


if __name__ == "__main__":
    n_timesteps = 1000000
    gamma = 0.99
    smoothing_window = 101
    n_repetitions = 5

    # hyperparameter_optimization(
    #     n_repetitions=n_repetitions,
    #     n_timesteps=n_timesteps,
    #     gamma=gamma,
    #     smoothing_window=smoothing_window,
    # )

    best_A2C(
        n_repetitions=n_repetitions,
        n_timesteps=n_timesteps,
        gamma=gamma,
        smoothing_window=smoothing_window,
    )
