# pyright: reportMissingImports=false
"""
Soft Actor-Critic (SAC) for the discrete CartPole-v1 action space.

Implementation follows Haarnoja et al., 2018 (https://arxiv.org/abs/1801.01290)
adapted to discrete actions (Christodoulou, 2019,
https://arxiv.org/abs/1910.07207). This file implements the *core* discrete
SAC algorithm: a stochastic policy, a soft Q-network with a Polyak-averaged
target, a fixed entropy temperature ``alpha``, and a replay buffer. No
additional engineering tricks are applied: no twin Q-networks (clipped
double-Q), no automatic entropy temperature tuning, no random warm-up phase,
and one gradient update per environment step.
"""

from Agent import BaseAgent

import math
import random
from collections import deque

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from Environment import CartPoleEnvironment
from tqdm import tqdm


class ReplayBuffer:
    """Simple FIFO replay buffer for SAC."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer: deque = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.stack(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.stack(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )


class SAC_Agent(BaseAgent):
    """Discrete Soft Actor-Critic agent (core formulation)."""

    def __init__(
        self,
        n_agent_state_elements,
        n_actions,
        actor_lr,
        critic_lr,
        gamma,
        tau,
        replay_buffer_size,
        batch_size,
        actor_hidden_nn,
        critic_hidden_nn,
        alpha: float = 1.0,
    ):
        self.n_agent_state_elements = n_agent_state_elements
        self.n_actions = n_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = int(batch_size)
        self.alpha = float(alpha)

        self.policy = PolicyNetwork(
            n_agent_state_elements,
            n_actions,
            actor_hidden_nn,
            activation=nn.ReLU,
        )
        self.q = QNetwork(
            n_agent_state_elements,
            n_actions,
            critic_hidden_nn,
            activation=nn.ReLU,
        )
        self.q_target = QNetwork(
            n_agent_state_elements,
            n_actions,
            critic_hidden_nn,
            activation=nn.ReLU,
        )
        self.q_target.load_state_dict(self.q.state_dict())
        for param in self.q_target.parameters():
            param.requires_grad = False

        self.opt_actor = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def select_action(self, obs, deterministic: bool = False):
        """Sample (or take argmax) an action from the soft policy."""

        state = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        with torch.no_grad():
            probs = self.policy(state)
            if deterministic:
                action = int(torch.argmax(probs, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(probs)
                action = int(dist.sample().item())
        return action, 0.0

    def _polyak_update(self, online: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for p_online, p_target in zip(online.parameters(), target.parameters()):
                p_target.data.mul_(1.0 - self.tau)
                p_target.data.add_(self.tau * p_online.data)

    def update(self, **kwargs):
        """Perform one SAC gradient step using a batch from the replay buffer."""

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        actions = actions.unsqueeze(-1)

        # Critic update
        with torch.no_grad():
            next_probs = self.policy(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q = self.q_target(next_states)
            # Soft value V(s') = sum_a pi(a|s') * (Q(s',a) - alpha * log pi(a|s'))
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_v

        q_pred = self.q(states).gather(1, actions).squeeze(-1)
        q_loss = F.mse_loss(q_pred, target_q)

        self.opt_q.zero_grad()
        q_loss.backward()
        self.opt_q.step()

        # Actor update
        probs = self.policy(states)
        log_probs = torch.log(probs + 1e-8)
        with torch.no_grad():
            q_cur = self.q(states)
        # J_pi = E_s [ sum_a pi(a|s) * (alpha * log pi(a|s) - Q(s,a)) ]
        actor_loss = (probs * (self.alpha * log_probs - q_cur)).sum(dim=-1).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # Soft target update
        self._polyak_update(self.q, self.q_target)

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


class QNetwork(nn.Module):
    """Q(s, ·): one output per discrete action."""

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
        return self.net(x)


def run_sac(
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
    eval_with_env_episode_trials: bool = True,
    n_eval_episodes: int = 5,
):
    """SAC training loop with eval-interval return recording.

    Steps through the environment one transition at a time, pushing into a
    replay buffer. Once the buffer has at least ``batch_size`` samples, one
    gradient update is performed per environment interaction.
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

    state, _ = env.reset()
    episode_steps = 0

    try:
        while global_step < n_timesteps:
            action, _ = agent.select_action(state, deterministic=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_done = bool(terminated or truncated)
            # Only flag a "true" terminal when the environment actually
            # terminated; truncation due to step limit must NOT zero the
            # bootstrap target.
            done_for_buffer = bool(terminated)

            agent.replay_buffer.push(state, action, reward, next_state, done_for_buffer)
            current_episode_return += float(reward)
            state = next_state
            global_step += 1
            episode_steps += 1

            if len(agent.replay_buffer) >= agent.batch_size:
                agent.update()

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
