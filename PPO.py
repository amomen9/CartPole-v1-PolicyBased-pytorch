# pyright: reportMissingImports=false
"""
Proximal Policy Optimisation (PPO-clipped) for CartPole-v1.

Implementation follows Schulman et al., 2017 (https://arxiv.org/abs/1707.06347).
The structure, training-loop signature, progress-bar handling and parallelisation
hooks mirror ``A2C.py`` so that PPO plugs into the same experiment pipeline.

Engineering tricks employed (all standard PPO additions on top of A2C):
- Rollout buffer of fixed length (``rollout_steps``) that may span multiple episodes.
- Generalised Advantage Estimation (GAE, Schulman et al., 2015) with
  ``gae_lambda`` for the advantage estimator A_t.
- Advantage normalisation (zero-mean / unit-variance) per rollout for variance
  reduction.
- Multiple gradient-ascent epochs over the same rollout with random mini-batches
  (``n_epochs`` × ``mini_batch_size``).
- Clipped surrogate objective ``L^CLIP``.
- Value-function clipping (Schulman et al., 2017 §5) bounding the TD target by
  ``clip_eps`` around the old value estimate.
- Entropy bonus to encourage exploration (``entropy_coef``).
- Global gradient-norm clipping (``max_grad_norm``).
"""

from Agent import BaseAgent

import math
import numpy as np
import torch
from torch import nn
from typing import Any, cast

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
        mini_batch_size,
        entropy_coef,
        value_coef,
        max_grad_norm,
        actor_hidden_nn,
        critic_hidden_nn,
    ):
        """Initialise the PPO agent."""

        self.n_agent_state_elements = n_agent_state_elements
        self.n_actions = n_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

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
        """Run several epochs of PPO-clipped updates on the collected rollout."""

        states = kwargs["states"]
        actions = kwargs["actions"]
        old_log_probs = kwargs["log_probs"]
        advantages = kwargs["advantages"]
        returns = kwargs["returns"]
        old_values = kwargs["values"]

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.long)
        old_log_probs_t = torch.tensor(np.array(old_log_probs), dtype=torch.float32)
        advantages_t = torch.tensor(np.array(advantages), dtype=torch.float32)
        returns_t = torch.tensor(np.array(returns), dtype=torch.float32)
        old_values_t = torch.tensor(np.array(old_values), dtype=torch.float32)

        # Advantage normalisation (per-rollout) for variance reduction.
        adv_std = advantages_t.std(unbiased=False)
        if torch.isfinite(adv_std) and adv_std.item() > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / (adv_std + 1e-8)

        n_samples = states_t.shape[0]
        mini_batch = min(int(self.mini_batch_size), n_samples) if n_samples > 0 else 0
        if mini_batch <= 0:
            return

        for _ in range(int(self.n_epochs)):
            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, mini_batch):
                idx = perm[start : start + mini_batch]

                probs = self.policy(states_t[idx])
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[idx])
                adv_batch = advantages_t[idx]
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.value_func(states_t[idx])
                # Clipped value loss (Schulman et al., 2017 §5)
                values_clipped = old_values_t[idx] + torch.clamp(
                    values_pred - old_values_t[idx],
                    -self.clip_eps,
                    self.clip_eps,
                )
                v_loss1 = (values_pred - returns_t[idx]) ** 2
                v_loss2 = (values_clipped - returns_t[idx]) ** 2
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                actor_loss = policy_loss - self.entropy_coef * entropy
                critic_loss = self.value_coef * value_loss

                self.opt_actor.zero_grad()
                actor_loss.backward()
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                self.opt_critic.zero_grad()
                critic_loss.backward()
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.value_func.parameters(), self.max_grad_norm)
                self.opt_critic.step()

    def evaluate(self, n_eval_episodes: int = 30, max_steps: int = 500) -> np.floating:
        """Evaluate the current policy greedily (deterministic).

        Uses environment episode rollouts (slower than the training-loop proxy
        ``last_episode_return``).
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
    full_episode_updates: bool = True,
):
    """PPO training loop with eval-interval return recording.

    Steps through the environment one transition at a time, filling a
    fixed-length rollout buffer (``rollout_steps``). Once full (or when the
    horizon is reached) PPO performs ``n_epochs`` mini-batch updates and the
    buffer is cleared. Progress handling mirrors the REINFORCE/A2C loops so
    parent-owned parallel tqdm bars behave consistently across algorithms.
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

            # Decide when to flush the buffer and update.
            # - full_episode_updates=True: update at the end of every episode
            #   using only that episode's trajectory.
            # - full_episode_updates=False: standard PPO with a fixed-length
            #   rollout buffer that may span multiple episodes.
            horizon_reached = global_step >= n_timesteps
            if full_episode_updates:
                should_update = episode_done or horizon_reached
            else:
                rollout_full = len(states) >= int(rollout_steps)
                should_update = rollout_full or horizon_reached

            if should_update and len(states) > 0:
                with torch.no_grad():
                    last_value_t = agent.value_func(state)
                last_value = float(last_value_t.item())
                # If episode just ended, terminal bootstrap is 0.
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
                    values=values,
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
