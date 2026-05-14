# pyright: reportMissingImports=false
"""
Soft Actor-Critic (SAC) for the discrete CartPole-v1 action space.

Implementation follows Haarnoja et al., 2018 (https://arxiv.org/abs/1801.01290)
and the discrete-action adaptation by Christodoulou, 2019
(https://arxiv.org/abs/1910.07207). The structure, training-loop signature,
progress-bar handling and parallelisation hooks mirror ``A2C.py`` so that SAC
plugs into the same experiment pipeline.

Engineering tricks employed (all standard SAC additions on top of A2C):
- Off-policy learning with an experience replay buffer.
- Twin Q-networks (clipped double Q-learning) to mitigate value overestimation
  (Fujimoto et al., 2018).
- Polyak (soft) target-network updates with rate ``tau``.
- Optional automatic entropy-temperature tuning (Haarnoja et al., 2019) with a
  learned ``log_alpha`` and the discrete-action target entropy
  ``target_entropy_ratio * log(n_actions)``. When ``auto_tune_alpha`` is
  ``False`` the original SAC formulation (Haarnoja et al., 2018) is used with
  a fixed ``alpha_init``.
- Random uniform warm-up phase (``warmup_steps``) before training kicks in.
- Mini-batch SGD via ``batch_size`` samples per gradient step and
  ``updates_per_step`` updates per environment step.
"""

from Agent import BaseAgent

import math
import random
from collections import deque
from typing import Any, cast

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
    """Discrete Soft Actor-Critic agent."""

    def __init__(
        self,
        n_agent_state_elements,
        n_actions,
        actor_lr,
        critic_lr,
        alpha_lr,
        gamma,
        tau,
        target_entropy_ratio,
        replay_buffer_size,
        batch_size,
        warmup_steps,
        updates_per_step,
        actor_hidden_nn,
        critic_hidden_nn,
        auto_tune_alpha: bool = True,
        alpha_init: float = 1.0,
    ):
        """Initialise the discrete SAC agent.

        When ``auto_tune_alpha`` is ``True`` the entropy temperature is learned
        (Haarnoja et al., 2019). Otherwise ``alpha`` is fixed to ``alpha_init``
        as in the original SAC (Haarnoja et al., 2018).
        """

        self.n_agent_state_elements = n_agent_state_elements
        self.n_actions = n_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.tau = tau
        self.target_entropy_ratio = target_entropy_ratio
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.updates_per_step = int(updates_per_step)
        self.auto_tune_alpha = bool(auto_tune_alpha)
        self.alpha_init = float(alpha_init)

        self.policy = PolicyNetwork(
            n_agent_state_elements,
            n_actions,
            actor_hidden_nn,
            activation=nn.ReLU,
        )
        self.q1 = QNetwork(
            n_agent_state_elements,
            n_actions,
            critic_hidden_nn,
            activation=nn.ReLU,
        )
        self.q2 = QNetwork(
            n_agent_state_elements,
            n_actions,
            critic_hidden_nn,
            activation=nn.ReLU,
        )
        self.q1_target = QNetwork(
            n_agent_state_elements,
            n_actions,
            critic_hidden_nn,
            activation=nn.ReLU,
        )
        self.q2_target = QNetwork(
            n_agent_state_elements,
            n_actions,
            critic_hidden_nn,
            activation=nn.ReLU,
        )
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False

        self.opt_actor = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.opt_q1 = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.opt_q2 = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)

        # For discrete actions, the target entropy is a fraction of the maximum
        # entropy ``log(n_actions)`` (Christodoulou, 2019). Only used when
        # ``auto_tune_alpha`` is ``True`` (Haarnoja et al., 2019).
        self.target_entropy = float(target_entropy_ratio) * math.log(n_actions)
        log_alpha_init = math.log(max(self.alpha_init, 1e-8))
        if self.auto_tune_alpha:
            self.log_alpha = torch.tensor([log_alpha_init], requires_grad=True)
            self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            # Fixed temperature: keep as a non-trainable tensor for a uniform
            # ``alpha`` property and no optimiser is created.
            self.log_alpha = torch.tensor([log_alpha_init], requires_grad=False)
            self.opt_alpha = None

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

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

        # ── Critic update ─────────────────────────────────────────────────
        with torch.no_grad():
            next_probs = self.policy(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q1 = self.q1_target(next_states)
            next_q2 = self.q2_target(next_states)
            next_q = torch.min(next_q1, next_q2)
            # Soft value V(s') = sum_a pi(a|s') * (Q(s',a) - alpha * log pi(a|s'))
            next_v = (next_probs * (next_q - self.alpha.detach() * next_log_probs)).sum(dim=-1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_v

        q1_pred = self.q1(states).gather(1, actions).squeeze(-1)
        q2_pred = self.q2(states).gather(1, actions).squeeze(-1)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.opt_q1.zero_grad()
        q1_loss.backward()
        self.opt_q1.step()

        self.opt_q2.zero_grad()
        q2_loss.backward()
        self.opt_q2.step()

        # ── Actor update ──────────────────────────────────────────────────
        probs = self.policy(states)
        log_probs = torch.log(probs + 1e-8)
        with torch.no_grad():
            q1_cur = self.q1(states)
            q2_cur = self.q2(states)
            q_cur = torch.min(q1_cur, q2_cur)
        # J_pi = E_s [ sum_a pi(a|s) * (alpha * log pi(a|s) - Q(s,a)) ]
        actor_loss = (probs * (self.alpha.detach() * log_probs - q_cur)).sum(dim=-1).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # ── Temperature (alpha) update ────────────────────────────────────
        if self.auto_tune_alpha:
            # Policy entropy estimate H_pi = - sum_a pi(a|s) * log pi(a|s)
            entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1)
            alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach()).mean()

            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()

        # ── Soft target update ────────────────────────────────────────────
        self._polyak_update(self.q1, self.q1_target)
        self._polyak_update(self.q2, self.q2_target)

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
    replay buffer. After ``warmup_steps`` random-action steps, ``updates_per_step``
    gradient updates are performed per environment interaction. Progress
    handling mirrors the A2C/REINFORCE loops so parent-owned parallel tqdm bars
    behave consistently across algorithms.
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
    n_actions = agent.n_actions

    try:
        while global_step < n_timesteps:
            if global_step < agent.warmup_steps:
                action = int(np.random.randint(0, n_actions))
            else:
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

            if global_step >= agent.warmup_steps and len(agent.replay_buffer) >= agent.batch_size:
                for _ in range(agent.updates_per_step):
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
