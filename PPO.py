# pyright: reportMissingImports=false
"""
Proximal Policy Optimisation (PPO-clipped) for CartPole-v1.

Implementation follows Schulman et al., 2017 (https://arxiv.org/abs/1707.06347)
with GAE (Schulman et al., 2015) as the advantage estimator, and includes the
full set of standard PPO engineering tricks (the "37 implementation details" /
Engstrom et al., 2020). Each trick is a constructor flag defaulting to its
optimal value, so the agent runs the strong configuration by default but every
trick can be ablated independently:

  - GAE advantage estimation (gae_lambda) with terminal bootstrap zeroing
  - per-minibatch advantage normalisation (normalize_advantages)
  - clipped surrogate policy objective (clip_eps)
  - value-function clipping (clip_vloss + clip_eps)
  - entropy bonus (entropy_coef)
  - value-loss coefficient (value_loss_coef)
  - several epochs over the rollout with shuffled mini-batches
    (n_epochs, num_minibatches)
  - global gradient-norm clipping (max_grad_norm)
  - optional KL-divergence early stopping (target_kl; off by default)
  - orthogonal initialisation with policy/value head gains, Adam eps,
    tanh activations and linear learning-rate annealing (anneal_lr)
"""

from Agent import BaseAgent

import math
import numpy as np
import torch
from torch import nn

from Environment import CartPoleEnvironment
import Library as fn
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
        # ── Engineering tricks (default-on / optimal) ──
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        adam_eps=1e-5,
        anneal_lr=True,
        orthogonal_init=True,
        normalize_advantages=True,
        normalize_obs=False,
        activation_name="tanh",
        num_minibatches=32,
        clip_vloss=True,
        target_kl=None,   # e.g. 0.015 to enable KL early-stopping (off by default)
    ):
        self.n_agent_state_elements = n_agent_state_elements
        self.n_actions = n_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs

        # Trick hyperparameters
        self.entropy_coef = float(entropy_coef)
        self.value_loss_coef = float(value_loss_coef)
        self.max_grad_norm = max_grad_norm
        self.anneal_lr = bool(anneal_lr)
        self.normalize_advantages = bool(normalize_advantages)
        self.normalize_obs = bool(normalize_obs)
        self.activation_name = activation_name
        self.num_minibatches = max(1, int(num_minibatches))
        self.clip_vloss = bool(clip_vloss)
        self.target_kl = target_kl
        self.base_actor_lr = float(actor_lr)
        self.base_critic_lr = float(critic_lr)

        self.policy = PolicyNetwork(
            n_agent_state_elements,
            n_actions,
            actor_hidden_nn,
            activation=activation_name,
            orthogonal_init=orthogonal_init,
            normalize_obs=normalize_obs,
        )
        self.value_func = ValueNetwork(
            n_agent_state_elements,
            critic_hidden_nn,
            activation=activation_name,
            orthogonal_init=orthogonal_init,
            normalize_obs=normalize_obs,
        )

        self.opt_actor = torch.optim.Adam(self.policy.parameters(), lr=actor_lr, eps=adam_eps)
        self.opt_critic = torch.optim.Adam(self.value_func.parameters(), lr=critic_lr, eps=adam_eps)

    def anneal_learning_rate(self, fraction_remaining):
        if not self.anneal_lr:
            return
        frac = max(0.0, float(fraction_remaining))
        for group in self.opt_actor.param_groups:
            group["lr"] = self.base_actor_lr * frac
        for group in self.opt_critic.param_groups:
            group["lr"] = self.base_critic_lr * frac

    def clip_gradients(self, parameters):
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)

    def select_action(self, obs):
        """Sample an action from the policy and return (action, log_prob, value)."""

        state = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        with torch.no_grad():
            logits = self.policy(state)
            value = self.value_func(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action).item(), float(value.item())

    def compute_gae(self, rewards, values, dones, last_value):
        """Generalised Advantage Estimation (Schulman et al., 2015).

        Returns '(advantages, returns)' where 'returns = advantages + values'.
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
        """Run several epochs of mini-batched PPO-clipped updates on the rollout.

        Engineering tricks applied here (all flag-controlled):
          - shuffled mini-batches reused for n_epochs passes (num_minibatches);
          - per-minibatch advantage normalisation (normalize_advantages);
          - clipped surrogate policy objective (clip_eps);
          - value-function clipping (clip_vloss);
          - entropy bonus (entropy_coef) and value-loss coef (value_loss_coef);
          - gradient-norm clipping on both networks (max_grad_norm);
          - optional KL-divergence early stopping (target_kl).
        """

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

        batch_size = states_t.shape[0]
        if batch_size == 0:
            return

        fn.update_obs_norm(self.policy, states_t)
        fn.update_obs_norm(self.value_func, states_t)

        # Old value predictions are needed for value-function clipping.
        with torch.no_grad():
            old_values_t = self.value_func(states_t)

        minibatch_size = max(1, batch_size // self.num_minibatches)
        stop_early = False

        for _ in range(int(self.n_epochs)):
            permutation = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                idx = permutation[start:start + minibatch_size]

                mb_states = states_t[idx]
                mb_actions = actions_t[idx]
                mb_old_log_probs = old_log_probs_t[idx]
                mb_advantages = advantages_t[idx]
                mb_returns = returns_t[idx]
                mb_old_values = old_values_t[idx]

                if self.normalize_advantages and mb_advantages.numel() > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std(unbiased=False) + 1e-8
                    )

                logits = self.policy(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                # Clipped surrogate policy objective with entropy bonus.
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # Value loss, optionally clipped around the old value prediction.
                values_pred = self.value_func(mb_states)
                if self.clip_vloss:
                    v_unclipped = (values_pred - mb_returns).pow(2)
                    v_clipped = mb_old_values + torch.clamp(
                        values_pred - mb_old_values, -self.clip_eps, self.clip_eps
                    )
                    v_clipped = (v_clipped - mb_returns).pow(2)
                    value_loss = self.value_loss_coef * torch.max(v_unclipped, v_clipped).mean()
                else:
                    value_loss = self.value_loss_coef * (values_pred - mb_returns).pow(2).mean()

                self.opt_actor.zero_grad()
                policy_loss.backward()
                self.clip_gradients(self.policy.parameters())
                self.opt_actor.step()

                self.opt_critic.zero_grad()
                value_loss.backward()
                self.clip_gradients(self.value_func.parameters())
                self.opt_critic.step()

            # KL early stopping: estimate KL once per epoch and bail if it grows
            # beyond the target (standard PPO safeguard against destructive steps).
            if self.target_kl is not None:
                with torch.no_grad():
                    logits = self.policy(states_t)
                    new_log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions_t)
                    approx_kl = (old_log_probs_t - new_log_probs).mean().item()
                if approx_kl > self.target_kl:
                    stop_early = True
            if stop_early:
                break

    def evaluate(self, n_eval_episodes: int = 30, max_steps: int = 500) -> np.floating:
        """Evaluate the current policy greedily (deterministic)."""
        env = CartPoleEnvironment(max_episode_length=max_steps, render_mode="rgb_array")

        total_returns: list[float] = []
        for _ in range(n_eval_episodes):
            obs, _ = env.reset()
            ep_return = 0.0

            for _ in range(max_steps):
                with torch.no_grad():
                    logits = self.policy(obs)
                    action = int(torch.argmax(logits, dim=-1).item())

                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)

                if terminated or truncated:
                    break

            total_returns.append(ep_return)

        env.close()
        return np.mean(total_returns)


class PolicyNetwork(nn.Module):
    def __init__(self, n_agent_state_elements, n_actions, hidden_layers, activation,
                 orthogonal_init=True, normalize_obs=False):
        super().__init__()
        activation = fn.resolve_activation(activation)

        layers = []
        input_size = n_agent_state_elements

        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_size, int(layer_size)))
            layers.append(activation())
            input_size = int(layer_size)

        layers.append(nn.Linear(input_size, n_actions))
        self.net = nn.Sequential(*layers)

        self.normalize_obs = bool(normalize_obs)
        if self.normalize_obs:
            fn.register_obs_norm(self, n_features=n_agent_state_elements)
        if orthogonal_init:
            fn.init_mlp_orthogonal(self.net, head_gain=0.01)  # policy head gain = 0.01

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return self.net(fn.normalize_obs(self, x))


class ValueNetwork(nn.Module):
    def __init__(self, n_agent_state_elements, hidden_layers, activation,
                 orthogonal_init=True, normalize_obs=False):
        super().__init__()
        activation = fn.resolve_activation(activation)

        layers = []
        input_size = n_agent_state_elements

        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_size, int(layer_size)))
            layers.append(activation())
            input_size = int(layer_size)

        layers.append(nn.Linear(input_size, 1))
        self.net = nn.Sequential(*layers)

        self.normalize_obs = bool(normalize_obs)
        if self.normalize_obs:
            fn.register_obs_norm(self, n_features=n_agent_state_elements)
        if orthogonal_init:
            fn.init_mlp_orthogonal(self.net, head_gain=1.0)  # value head gain = 1.0

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return self.net(fn.normalize_obs(self, x)).squeeze(-1)


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
    fixed-length rollout buffer ('rollout_steps'). Once full PPO performs
    'n_epochs' full-batch gradient updates and the buffer is cleared.
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

            # Linear learning-rate annealing over the full training horizon.
            agent.anneal_learning_rate(1.0 - global_step / n_timesteps)

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
