# pyright: reportMissingImports=false
"""
Proximal Policy Optimisation (PPO-clipped) for CartPole-v1.

Implementation follows Schulman et al., 2017 (https://arxiv.org/abs/1707.06347).
This file implements the *core* PPO-clipped algorithm with GAE
(Schulman et al., 2015) as the advantage estimator. No additional engineering
tricks are applied: no advantage normalisation, no value-function clipping,
no entropy bonus, no gradient-norm clipping, no mini-batching (each epoch
runs one full-batch gradient step over the rollout).
"""

from Agent import BaseAgent

import math
import numpy as np
import torch
from torch import nn

from Environment import CartPoleEnvironment
from tqdm import tqdm


class PPO_Agent(BaseAgent):
    """PPO-clipped agent for a discrete action space (categorical policy)."""

    def __init__(
        self,
        n_agent_state_elements,
        n_actions,
        actor_lr,
        critic_lr,
        gamma,
        gae_lambda,
        clip_eps,
        n_epochs,
        actor_hidden_nn,
        critic_hidden_nn,
    ):
        self.n_agent_state_elements = n_agent_state_elements
        self.n_actions = n_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs

        self.policy = PolicyNetwork(
            n_agent_state_elements,
            n_actions,
            actor_hidden_nn,
            activation=nn.Tanh,
        )
        self.value_func = ValueNetwork(
            n_agent_state_elements,
            critic_hidden_nn,
            activation=nn.Tanh,
        )

        self.opt_actor = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.opt_critic = torch.optim.Adam(self.value_func.parameters(), lr=critic_lr)

    def select_action(self, obs):
        """Sample an action from the policy and return (action, log_prob, value)."""

        state = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        with torch.no_grad():
            probs = self.policy(state)
            value = self.value_func(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action).item(), float(value.item())

    def compute_gae(self, rewards, values, dones, last_value):
        """Generalised Advantage Estimation (Schulman et al., 2015).

        Returns ``(advantages, returns)`` where ``returns = advantages + values``.
        """

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = float(last_value)
        for t in reversed(range(T)):
            non_terminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + np.asarray(values, dtype=np.float32)
        return advantages, returns

    def update(self, **kwargs):
        """Run several full-batch epochs of PPO-clipped updates on the rollout."""

        states = kwargs["states"]
        actions = kwargs["actions"]
        old_log_probs = kwargs["log_probs"]
        advantages = kwargs["advantages"]
        returns = kwargs["returns"]

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.long)
        old_log_probs_t = torch.tensor(np.array(old_log_probs), dtype=torch.float32)
        advantages_t = torch.tensor(np.array(advantages), dtype=torch.float32)
        returns_t = torch.tensor(np.array(returns), dtype=torch.float32)

        if states_t.shape[0] == 0:
            return

        for _ in range(int(self.n_epochs)):
            probs = self.policy(states_t)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_t)

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            values_pred = self.value_func(states_t)
            value_loss = 0.5 * (values_pred - returns_t).pow(2).mean()

            self.opt_actor.zero_grad()
            policy_loss.backward()
            self.opt_actor.step()

            self.opt_critic.zero_grad()
            value_loss.backward()
            self.opt_critic.step()

    def evaluate(self, n_eval_episodes: int = 30, max_steps: int = 500) -> np.floating:
        """Evaluate the current policy greedily (deterministic)."""
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


def run_ppo(
    agent,
    env,
    n_timesteps=1000000,
    eval_interval=250,
    truncation_step=500,
    rollout_steps=2048,
    enable_progress_bar=True,
    progress_bar_desc="Env Steps",
    progress_bar_position=None,
    shared_step_counter=None,
    max_eval_episode_length=None,
    eval_with_env_episode_trials: bool = True,
    n_eval_episodes: int = 5,
):
    """PPO training loop with eval-interval return recording.

    Steps through the environment one transition at a time, filling a
    fixed-length rollout buffer (``rollout_steps``). Once full PPO performs
    ``n_epochs`` full-batch gradient updates and the buffer is cleared.
    """

    data_count = math.ceil(n_timesteps / eval_interval)
    eval_returns = np.empty(data_count, dtype=np.float32)
    eval_timesteps = np.empty(data_count, dtype=np.int32)
    eval_write_idx = 0

    global_step = 0
    last_episode_return = 0.0
    current_episode_return = 0.0

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

    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
    state, _ = env.reset()
    episode_steps = 0

    try:
        while global_step < n_timesteps:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_done = bool(terminated or truncated)

            states.append(np.asarray(state, dtype=np.float32))
            actions.append(int(action))
            rewards.append(float(reward))
            log_probs.append(float(log_prob))
            values.append(float(value))
            dones.append(episode_done)

            current_episode_return += float(reward)
            state = next_state
            global_step += 1
            episode_steps += 1

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

            if episode_done or episode_steps >= truncation_step:
                last_episode_return = current_episode_return
                if pbar is not None:
                    pbar.set_postfix_str(
                        f"episode_reward={last_episode_return:.2f}", refresh=False
                    )
                current_episode_return = 0.0
                episode_steps = 0
                state, _ = env.reset()

            horizon_reached = global_step >= n_timesteps
            rollout_full = len(states) >= int(rollout_steps)
            should_update = rollout_full or horizon_reached

            if should_update and len(states) > 0:
                with torch.no_grad():
                    last_value_t = agent.value_func(state)
                last_value = float(last_value_t.item())
                if dones[-1]:
                    last_value = 0.0

                advantages, returns_buf = agent.compute_gae(
                    rewards=rewards,
                    values=values,
                    dones=dones,
                    last_value=last_value,
                )
                agent.update(
                    states=states,
                    actions=actions,
                    log_probs=log_probs,
                    advantages=advantages,
                    returns=returns_buf,
                )
                states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
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
