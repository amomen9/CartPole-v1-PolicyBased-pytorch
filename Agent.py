
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import torch
import torch.nn as nn
import Environment
import Library as fn




# Begin Class BaseAgent ##########################################################################
class BaseAgent:
    """Base agent for policy-gradient methods on CartPole.

    Subclasses (REINFORCE, AC, A2C, A3C) must override 'update()'.

    The actor is a Policy_NN (sigmoid curve output → P(action=1|s)).
    The critic is a Value_NN:
        - V_phi(s) for advantage-based methods (AC, A2C, A3C)
        - Q_phi(s) when an explicit Q critic is desired (Like DQN - Optional for this project)
    """

    def __init__(self, actor_hidden_nn=np.array([16, 16]),
                 critic_hidden_nn=np.array([64, 64]),
                 actor_lr=0.001, critic_lr=0.001,
                 gamma=0.99, use_critic=False, critic_type='V',
                 # ── Engineering tricks (default to their optimal/ON values) ──
                 entropy_coef=0.01,        # weight of the entropy bonus -coef*H(pi)
                 value_loss_coef=0.5,      # weight of the critic regression loss
                 max_grad_norm=0.5,        # global grad-norm clip (None disables)
                 adam_eps=1e-5,            # Adam epsilon (PPO-standard, more stable)
                 anneal_lr=True,           # linearly anneal LR to 0 over training
                 orthogonal_init=True,     # orthogonal weight init with head gains
                 normalize_advantages=True,# whiten advantages/returns per batch
                 normalize_obs=False):     # running obs-normalization (opt-in; see note)
        # Actor (policy network pi_theta)
        self.actor = fn.Policy_NN(
            nn_hidden_layer_widths=actor_hidden_nn,
            orthogonal_init=orthogonal_init,
            normalize_obs=normalize_obs,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=adam_eps)

        # Do not use critic for REINFORCE (use_critic=False), but set up critic network and optimizer if critic methods are used (AC, A2C, A3C - use_critic=True).
        self.use_critic = use_critic
        # The critic_type determines whether it's a state-value (V) or action-value (Q) critic.
        self.critic_type = critic_type  # 'V' for state-value (AC, A2C, A3C), 'Q' for action-value (explicit Q critic)

        if use_critic:
            # Critic (V_phi or Q_phi)
            output_size = 1 if critic_type == 'V' else 2  # Q outputs one value per action
            self.critic = fn.Value_NN(
                nn_hidden_layer_widths=critic_hidden_nn,
                output_size=output_size,
                orthogonal_init=orthogonal_init,
                normalize_obs=normalize_obs,
            )
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=adam_eps)
        else:
            self.critic = None
            self.critic_optimizer = None

        self.gamma = gamma

        # ── store trick hyperparameters ──
        self.entropy_coef = float(entropy_coef)
        self.value_loss_coef = float(value_loss_coef)
        self.max_grad_norm = max_grad_norm
        self.anneal_lr = bool(anneal_lr)
        self.orthogonal_init = bool(orthogonal_init)
        self.normalize_advantages = bool(normalize_advantages)
        self.normalize_obs = bool(normalize_obs)
        # Base learning rates remembered so the LR scheduler can rescale from a
        # fixed reference instead of compounding each step.
        self.base_actor_lr = float(actor_lr)
        self.base_critic_lr = float(critic_lr)

    # ── Shared engineering-trick helpers ─────────────────────────────────────
    def anneal_learning_rate(self, fraction_remaining):
        """Linearly scale every optimizer LR by `fraction_remaining` in [0, 1].

        Called by each training loop once per environment step using
        fraction_remaining = 1 - global_step / n_timesteps. No-op when
        anneal_lr is disabled.
        """
        if not self.anneal_lr:
            return
        frac = max(0.0, float(fraction_remaining))
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.base_actor_lr * frac
        if self.critic_optimizer is not None:
            for group in self.critic_optimizer.param_groups:
                group["lr"] = self.base_critic_lr * frac

    def clip_gradients(self, parameters):
        """Apply global gradient-norm clipping when max_grad_norm is set."""
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)

    @staticmethod
    def whiten(x, eps=1e-8):
        """Zero-mean / unit-std normalization with a near-constant guard."""
        std = x.std(unbiased=False)
        if not torch.isfinite(std) or std < eps:
            return x - x.mean()
        return (x - x.mean()) / (std + eps)

    def policy_distribution(self, states_tensor):
        """Bernoulli policy distribution over the single CartPole action logit."""
        logits = self.actor(states_tensor).squeeze(-1)
        return torch.distributions.Bernoulli(logits=logits)

    def select_action(self, obs):
        """Sample action from pi_theta(s). Returns (action, log_prob)."""
        state = torch.as_tensor(obs, dtype=torch.float32)
        logit = self.actor(state)
        prob = torch.sigmoid(logit)
        dist = torch.distributions.Bernoulli(probs=prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze()

    ###### !!! Only for critic-based methods (AC, A2C, A3C) !!! ######
    def get_value(self, obs):  # Get V_phi(s) or Q_phi(s) from the critic network, depending on critic_type. Returns a scalar value (V) or a vector of action values (Q). If no critic is used, returns 0.0.
        """Get either V_phi(s) or Q_phi(s) from the critic network, depending on critic_type. Returns a scalar value (V) or a vector of action values (Q). If no critic is used, returns 0.0."""
        if self.critic is None:
            return torch.tensor(0.0)
        state = torch.as_tensor(obs, dtype=torch.float32)
        return self.critic(state).squeeze()

    ### Every method (REINFORCE, AC, A2C, A3C) must implement its own update() method ###
    def update(self, **kwargs):
        raise NotImplementedError('Subclasses must implement their specific update method')

    def compute_discounted_returns(self, rewards):
        """Compute G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k} for each timestep."""
        T = len(rewards)
        returns = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):    # reversed, so that the latter is the reward, the more it would be multiplied by the discount factor, so that rewards in the far future have less impact on the return than rewards in the near future.
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def evaluate(self, n_eval_episodes=30, max_steps=500):
        """Evaluate current policy greedily (deterministic: P >= 0.5 → action 1)."""
        # create a new environment instance for evaluation (to avoid interfering with training env). Set render_mode to None for faster evaluation. You can set it to "rgb_array" if you want to visualize the evaluation episodes (but this will be slower).
        env= Environment.CartPoleEnvironment(max_episode_length=max_steps, render_mode=None)  # Create a new environment instance for evaluation (to avoid interfering with training env). Set render_mode to None for faster evaluation. You can set it to "rgb_array" if you want to visualize the evaluation episodes (but this will be slower). 
        total_returns = []
        for _ in range(n_eval_episodes):
            obs, _ = env.reset()
            ep_return = 0.0
            for _ in range(max_steps):
                with torch.no_grad():
                    state = torch.as_tensor(obs, dtype=torch.float32)
                    logit = self.actor(state)
                    prob = torch.sigmoid(logit).item()
                action = 1 if prob >= 0.5 else 0
                obs, reward, done, truncated, info = env.step(action)
                ep_return += reward
                if done or truncated:
                    break
            total_returns.append(ep_return)
        return np.mean(total_returns)

# End Class BaseAgent ##########################################################################


def trained_nn_policy(model, obs):
    """Use a trained Policy_NN to select an action (for visualization)."""
    with torch.no_grad():
        state = torch.as_tensor(obs, dtype=torch.float32)
        logit = model(state)
        prob = torch.sigmoid(logit).item()
    return int(prob >= 0.5)










