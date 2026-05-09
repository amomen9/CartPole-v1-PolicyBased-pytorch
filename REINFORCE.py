import math
import torch
import numpy as np
from tqdm import tqdm
from Agent import BaseAgent

class REINFORCE_Agent(BaseAgent):

    def update(self, **kwargs):
        """REINFORCE policy gradient update."""
        log_probs = kwargs['log_probs']
        rewards = kwargs['rewards']
        G = self.compute_discounted_returns(rewards)
        G = (G - G.mean()) / (G.std(unbiased=False) + 1e-8)

        policy_loss = torch.stack([-lp * g for lp, g in zip(log_probs, G)]).sum()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()


def run_reinforce(
    agent: REINFORCE_Agent,
    env,
    n_timesteps=1000000,
    eval_interval=250,
    truncation_step=500,
    max_eval_episode_length=None,
    enable_progress_bar=True,
    progress_bar_desc="Env Steps",
    progress_bar_position=None,
    shared_step_counter=None,
    eval_with_env_episode_trials: bool = False,
    n_eval_episodes: int = 5,
):
    """REINFORCE training loop. Steps through the environment one step at a time.

    Every eval_interval global steps, records either:
    - the fast proxy `last_episode_return` (default), or
    - a greedy environment evaluation via `agent.evaluate()` ("episode trials").

    Returns (eval_returns, eval_timesteps) as pre-allocated numpy arrays, matching DQN data collection.

    If *shared_step_counter* (a multiprocessing.Value) is provided, the loop
    writes global_step into it periodically so a parent process can monitor
    progress. When a shared counter is used, enable_progress_bar is typically
    False (the parent owns the tqdm bars).
    """
    data_count = math.ceil(n_timesteps / eval_interval)
    eval_returns = np.empty(data_count, dtype=np.float32)
    eval_timesteps = np.empty(data_count, dtype=np.int32)
    eval_write_idx = 0

    global_step = 0
    last_episode_return = 0.0

    if max_eval_episode_length is None:
        max_eval_episode_length = truncation_step

    # ── tqdm progress bar (matching assignment2_repo DQN pattern) ──
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
            # --- collect one full episode step by step ---
            env.reset()
            obs = env.obs
            log_probs, episode_rewards = [], []
            done, truncated = False, False

            for _ in range(truncation_step):
                action, log_prob = agent.select_action(obs)
                next_obs, reward, done, truncated, info = env.step(action)

                log_probs.append(log_prob)
                episode_rewards.append(reward)
                global_step += 1
                obs = next_obs

                # Update progress bar every 512 steps (reduces terminal spam)
                if (global_step - last_progress_update) >= 512 or global_step >= n_timesteps:
                    if pbar is not None:
                        pbar.update(global_step - last_progress_update)
                    if shared_step_counter is not None:
                        shared_step_counter.value = global_step
                    last_progress_update = global_step

                # Record eval snapshot at every eval_interval steps
                if global_step % eval_interval == 0:
                    if eval_with_env_episode_trials:
                        # More accurate evaluation (significantly slower)
                        eval_returns[eval_write_idx] = agent.evaluate(
                            n_eval_episodes=n_eval_episodes,
                            max_steps=max_eval_episode_length,
                        )
                    else:
                        # Fast proxy: last completed episode return (updated after episode end)
                        eval_returns[eval_write_idx] = last_episode_return
                    eval_timesteps[eval_write_idx] = global_step
                    eval_write_idx += 1


                if done or truncated or global_step >= n_timesteps:
                    break
            # Update policy on the completed episode
            last_episode_return = sum(episode_rewards)
            agent.update(log_probs=log_probs, rewards=episode_rewards)

            if pbar is not None:
                pbar.set_postfix_str(f"episode_reward={last_episode_return:.2f}", refresh=False)

        # Final snapshot if the last step didn't land on an eval boundary
        if global_step % eval_interval != 0:
            eval_returns[eval_write_idx] = last_episode_return
            eval_timesteps[eval_write_idx] = global_step
            eval_write_idx += 1
    finally:
        if shared_step_counter is not None:
            shared_step_counter.value = global_step
        if pbar is not None:
            pbar.close()
        env.close()

    return eval_returns[:eval_write_idx], eval_timesteps[:eval_write_idx]
