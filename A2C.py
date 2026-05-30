# pyright: reportMissingImports=false
from Agent import BaseAgent

import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Any, cast

from Environment import CartPoleEnvironment
from Library import average_over_repetitions, LearningCurvePlot, smooth
import Library as fn
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
        use_gae=True,
        gae_lambda=0.95,
    ):
        """Initialise the A2C agent."""

        self.n_agent_state_elements = n_agent_state_elements
        self.n_actions = n_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.TN_step = TN_step

        # Trick hyperparameters
        self.entropy_coef = float(entropy_coef)
        self.value_loss_coef = float(value_loss_coef)
        self.max_grad_norm = max_grad_norm
        self.anneal_lr = bool(anneal_lr)
        self.normalize_advantages = bool(normalize_advantages)
        self.normalize_obs = bool(normalize_obs)
        self.activation_name = activation_name
        self.use_gae = bool(use_gae)
        self.gae_lambda = float(gae_lambda)
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

    # ── trick helpers ──
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

    @staticmethod
    def whiten(x, eps=1e-8):
        std = x.std(unbiased=False)
        if not torch.isfinite(std) or std < eps:
            return x - x.mean()
        return (x - x.mean()) / (std + eps)

    def select_action(self, obs):
        """Sample an action from the policy."""

        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    def compute_returns(self, rewards, next_states, dones):
        """Compute the n-step return estimate for each timestep (bootstrapped V)."""

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

    def compute_gae(self, states_t, rewards, next_states, dones):
        """Generalised Advantage Estimation (Schulman et al., 2015).

        Used in place of the fixed n-step return when use_gae is True; returns
        '(advantages, value-targets)' both detached for use as training targets.
        """
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        with torch.no_grad():
            values = self.value_func(states_t)
            next_values = self.value_func(next_states_t)

        T = len(rewards)
        advantages = torch.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            non_terminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * float(next_values[t]) * non_terminal - float(values[t])
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, **kwargs):
        """Update policy and value networks from one trajectory.

        Engineering tricks: optional GAE advantage estimator (use_gae), advantage
        whitening (normalize_advantages), entropy bonus (entropy_coef), critic
        loss scaling (value_loss_coef) and gradient-norm clipping (max_grad_norm).
        """

        states = kwargs["states"]
        actions = kwargs["actions"]
        rewards = kwargs["rewards"]
        next_states = kwargs["next_states"]
        dones = kwargs["dones"]

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.long)
        fn.update_obs_norm(self.policy, states_t)
        fn.update_obs_norm(self.value_func, states_t)

        # Advantage + critic target: GAE or fixed n-step return.
        if self.use_gae:
            advantages, returns = self.compute_gae(states_t, rewards, next_states, dones)
        else:
            returns = self.compute_returns(rewards, next_states, dones)
            with torch.no_grad():
                advantages = returns - self.value_func(states_t)
        advantages = advantages.detach()
        returns = returns.detach()

        # Critic regression V(s) -> returns
        v_s = self.value_func(states_t)
        critic_loss = self.value_loss_coef * ((returns - v_s) ** 2).sum()
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.clip_gradients(self.value_func.parameters())
        self.opt_critic.step()

        # Actor: advantage-weighted log-prob + entropy bonus
        adv = self.whiten(advantages) if self.normalize_advantages else advantages
        logits = self.policy(states_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy_term = dist.entropy().sum()
        actor_loss = -(adv * log_probs).sum() - self.entropy_coef * entropy_term

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.clip_gradients(self.policy.parameters())
        self.opt_actor.step()

    def evaluate(self, n_eval_episodes: int = 30, max_steps: int = 500) -> np.floating:
        """Evaluate the current policy greedily (deterministic).

        Uses environment episode rollouts (slower than the training-loop proxy
        'last_episode_return').
        """
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
