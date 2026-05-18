import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
from tqdm import tqdm
from .Environment import make_env, env, max_episode_length, basic_policy, show_one_episode

_DQN_ONLINE_MODEL_REF = None
_DQN_TARGET_MODEL = None




def reset_dqn_target_cache():
    """Reset cached target-network references before a new trial starts."""
    global _DQN_ONLINE_MODEL_REF, _DQN_TARGET_MODEL
    _DQN_ONLINE_MODEL_REF = None
    _DQN_TARGET_MODEL = None

################[ PolicyNetwork Class              ]################
def softmax(x, temp):   # aka Boltzmann policy (Mentioned in Assignment 1 as Boltzmann in the assignment)
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax
    probs = np.exp(z)/np.sum(np.exp(z)) # compute softmax
    selected_action = np.random.choice(len(x), p=probs) # Sample action from
    return int(selected_action)
####################################################################

################[ PolicyNetwork Class              ]################
class PolicyNetwork(nn.Module):
    def __init__(self, nn_depth=3, nn_hidden_layer_widths=np.array([5])):     # nn_depth is the total number of layers (input + hidden + output), and nn_hidden_layer_widths are the input and output sizes of all layers in depth order (the first number is for the input and the last number is for the output). For example, if nn_depth=3 and nn_hidden_layer_widths=[5, 5], then the network will have an input layer of size 4 (the state dimension), a hidden layer of size 5, another hidden layer of size 5, and an output layer of size 2 (the action dimension).
        super().__init__()
        hidden_widths = np.asarray(nn_hidden_layer_widths, dtype=np.int32).tolist()
        if len(hidden_widths) == 0:
            raise ValueError("nn_hidden_layer_widths must contain at least one hidden-layer width")

        layers = []
        in_features = 4
        for width in hidden_widths:
            layers.append(nn.Linear(in_features, int(width)))
            layers.append(nn.ReLU())
            in_features = int(width)
        layers.append(nn.Linear(in_features, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)
####################################################################

################[ Choose Action Function           ]################
def choose_action(model, obs, exploration_method="epsilon_greedy", epsilon=0.0, temp=1.0): # For CartPole, this actions would be 0 or 1, but could be more for other environments.
    state = torch.as_tensor(obs, dtype=torch.float32)
    with torch.inference_mode():
        q_values = model(state)     # the neural network output of the Q-Values Q[s,0] and Q[s,1] for the current state.

    if exploration_method == "epsilon_greedy":
        if random.random() < epsilon:
            return random.randrange(q_values.shape[-1])
        return int(q_values.argmax().item())

    if exploration_method == "softmax":
        if temp is None or temp <= 0:
            raise ValueError("temp must be > 0 for softmax exploration")
        return softmax(q_values.detach().cpu().numpy(), temp)

    raise ValueError("exploration_method must be either 'epsilon_greedy' or 'softmax'")
####################################################################

################[ DQN Update Function              ]################
def dqn_update(model, optimizer, state, action, reward, next_state, terminated, discount_factor,
               target_network_step=1, global_step=None, target_network_active=True):
    global _DQN_ONLINE_MODEL_REF, _DQN_TARGET_MODEL

    if target_network_active and target_network_step < 1:
        raise ValueError("target_network_step must be >= 1")

    online_base_model = model
    if target_network_active:
        # Keep an internal target network and refresh it only at the requested interval.
        if _DQN_ONLINE_MODEL_REF is not online_base_model or _DQN_TARGET_MODEL is None:
            _DQN_ONLINE_MODEL_REF = online_base_model
            _DQN_TARGET_MODEL = copy.deepcopy(online_base_model)
            _DQN_TARGET_MODEL.eval()

    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)
    reward_tensor = torch.tensor(reward, dtype=torch.float32)

    current_q = model(state_tensor)[action]

    if target_network_active:
        should_sync_target = (global_step is None) or (global_step % target_network_step == 0)
        if should_sync_target:
            assert _DQN_TARGET_MODEL is not None
            _DQN_TARGET_MODEL.load_state_dict(online_base_model.state_dict())
        bootstrap_model = _DQN_TARGET_MODEL
    else:
        bootstrap_model = model

    with torch.no_grad():
        next_q = bootstrap_model(next_state_tensor).max()
        target_q = reward_tensor + discount_factor * next_q * (1 - int(terminated))

    loss = F.mse_loss(current_q, target_q)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()
####################################################################

################[ DQN Batch Update (Experience Replay) ]############
def dqn_batch_update(model, optimizer, replay_buffer, batch_size, discount_factor,
                     target_network_step=1, global_step=None, target_network_active=True):
    """Sample a batch from replay buffer and perform a vectorised Q-learning update."""
    global _DQN_ONLINE_MODEL_REF, _DQN_TARGET_MODEL

    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, terminateds = zip(*batch)

    states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
    next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32)
    terminateds_t = torch.as_tensor(terminateds, dtype=torch.float32)

    online_base_model = model
    if target_network_active:
        if target_network_step < 1:
            raise ValueError("target_network_step must be >= 1")
        # Reuse the target network managed by dqn_update.
        if _DQN_ONLINE_MODEL_REF is not online_base_model or _DQN_TARGET_MODEL is None:
            _DQN_ONLINE_MODEL_REF = online_base_model
            _DQN_TARGET_MODEL = copy.deepcopy(online_base_model)
            _DQN_TARGET_MODEL.eval()

        should_sync = (global_step is None) or (global_step % target_network_step == 0)
        if should_sync:
            assert _DQN_TARGET_MODEL is not None
            _DQN_TARGET_MODEL.load_state_dict(online_base_model.state_dict())
        bootstrap_model = _DQN_TARGET_MODEL
    else:
        bootstrap_model = model

    current_q = model(states_t).gather(1, actions_t).squeeze(1)

    with torch.no_grad():
        next_q = bootstrap_model(next_states_t).max(dim=1).values
        target_q = rewards_t + discount_factor * next_q * (1 - terminateds_t)

    loss = F.mse_loss(current_q, target_q)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()
####################################################################

################[ Train Naive DQN Function         ]################
def train_dqn(model, optimizer, env, n_env_steps, discount_factor,
                    epsilon_start: float | None = 1.0, epsilon_end: float | None = 0.05, epsilon_decay: float | None = 0.995,
                    target_network_step=1, max_episode_length=None, max_eval_episode_length=None, n_returns_interval=250,
                    target_network_active=True,
                    epsilon_decay_interval=1, exploration_method="epsilon_greedy",
                    softmax_temp: float | None = 1.0, eval_with_env_episode_trials: bool = False, n_eval_episodes: int = 5, enable_progress_bar=True,
                    progress_bar_position=None, progress_bar_desc="Env Steps",
                    shared_step_counter=None,
                    er_active=False, er_replay_buffer_size=10000,
                    er_batch_size=64, er_min_replay_size=100,
                    er_sample_train_frequency=1, er_replay_ratio=1.0,
                    full_episode_updates: bool = False):
    if n_env_steps < 1:
        raise ValueError("n_env_steps must be >= 1")
    if max_episode_length is not None and max_episode_length < 1:
        raise ValueError("max_episode_length must be >= 1")
    if n_returns_interval < 1:
        raise ValueError("n_returns_interval must be >= 1")
    if epsilon_decay_interval < 1:
        raise ValueError("epsilon_decay_interval must be >= 1")
    if exploration_method not in ("epsilon_greedy", "softmax"):
        raise ValueError("exploration_method must be either 'epsilon_greedy' or 'softmax'")
    if exploration_method == "epsilon_greedy":
        if any(v is None for v in (epsilon_start, epsilon_end, epsilon_decay)):
            raise ValueError("epsilon_start, epsilon_end, epsilon_decay must be provided for epsilon_greedy exploration")
        assert epsilon_start is not None
        assert epsilon_end is not None
        assert epsilon_decay is not None
    if exploration_method == "softmax" and (softmax_temp is None or softmax_temp <= 0):
        raise ValueError("softmax_temp must be > 0 for softmax exploration")
    if exploration_method == "softmax":
        assert softmax_temp is not None
    epsilon_end_value = float(epsilon_end) if epsilon_end is not None else 0.0
    epsilon_decay_value = float(epsilon_decay) if epsilon_decay is not None else 1.0
    softmax_temp_value = float(softmax_temp) if softmax_temp is not None else 1.0
    data_count = math.ceil(n_env_steps / n_returns_interval)
    eval_returns = np.empty(data_count, dtype=np.float32)
    eval_time_steps = np.empty(data_count, dtype=np.int32)
    eval_write_idx = 0
    epsilon = float(epsilon_start) if epsilon_start is not None else 0.0
    global_step = 0
    model.train()
    episode = 1
    last_episode_return = 0.0          # tracks the return of the most recently completed episode

    replay_buffer: ReplayBuffer | None = None
    if er_active:
        replay_buffer = ReplayBuffer(er_replay_buffer_size)

    # In full-episode-update mode, transitions and update triggers are
    # accumulated during the episode and flushed when it ends.
    pending_no_er_transitions: list = []
    pending_er_updates: int = 0

    train_env = env if max_episode_length is None else make_env(max_episode_length=max_episode_length)

    if max_eval_episode_length is None:
        max_eval_episode_length = max_episode_length

    eval_env = None
    if bool(eval_with_env_episode_trials):
        eval_env = make_env(max_episode_length=max_eval_episode_length, render_mode=None)

    # Use autoreset so training can continue as a single step stream.
    autoreset_cls = getattr(gym.wrappers, "Autoreset", None)
    if autoreset_cls is None:
        autoreset_cls = getattr(gym.wrappers, "AutoResetWrapper", None)
    if autoreset_cls is None:
        raise RuntimeError("No autoreset wrapper found in gymnasium.wrappers")
    assert autoreset_cls is not None
    step_env = autoreset_cls(train_env)

    seed = torch.randint(0, 2**32, size=()).item()
    obs, info = step_env.reset(seed=seed)
    total_reward = 0.0
    pbar = None
    if enable_progress_bar:
        tqdm_kwargs = {
            "total": n_env_steps,
            "desc": progress_bar_desc,
            "unit": "step",
            "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            "dynamic_ncols": True,
            "leave": True,
        }
        if progress_bar_position is not None:
            tqdm_kwargs["position"] = int(progress_bar_position)
        pbar = tqdm(
            **tqdm_kwargs,
        )
    last_progress_update = 0

    try:
        while global_step < n_env_steps:
            if exploration_method == "softmax":
                action = choose_action(
                    model,
                    obs,
                    exploration_method=exploration_method,
                    epsilon=epsilon,
                    temp=softmax_temp_value,
                )
            else:
                action = choose_action(
                    model,
                    obs,
                    exploration_method=exploration_method,
                    epsilon=epsilon,
                    temp=1.0,
                )
            next_obs, reward, terminated, truncated, _info = step_env.step(action)
            global_step += 1
            if not er_active:
                if full_episode_updates:
                    # Defer the per-step Q-learning update to episode end.
                    pending_no_er_transitions.append(
                        (obs, action, reward, next_obs, bool(terminated))
                    )
                else:
                    dqn_update(
                        model, optimizer, obs, action, reward, next_obs, terminated, discount_factor,
                        target_network_step=target_network_step,
                        global_step=global_step,
                        target_network_active=bool(target_network_active),
                    )
            # Also Experience Replay in the training:
            if er_active:
                assert replay_buffer is not None
                replay_buffer.push(obs, action, reward, next_obs, terminated)
                if len(replay_buffer) >= er_min_replay_size and global_step % er_sample_train_frequency == 0:
                    if full_episode_updates:
                        # Defer the ER batch updates to episode end (same total count).
                        pending_er_updates += int(er_replay_ratio)
                    else:
                        for _ in range(int(er_replay_ratio)):
                            dqn_batch_update(model, optimizer, replay_buffer, er_batch_size, discount_factor,
                                             target_network_step=target_network_step,
                                             global_step=global_step,
                                             target_network_active=bool(target_network_active))
            # Experience Replay end
            obs = next_obs
            total_reward += reward
            if exploration_method == "epsilon_greedy" and global_step % epsilon_decay_interval == 0:
                epsilon = max(epsilon_end_value, epsilon * epsilon_decay_value)

            # Reduce terminal spam by updating the progress bar every 512 env steps.
            if (global_step - last_progress_update) >= 512 or global_step == n_env_steps:
                if pbar is not None:
                    pbar.update(global_step - last_progress_update)
                if shared_step_counter is not None:
                    shared_step_counter.value = global_step
                last_progress_update = global_step

            if terminated or truncated:
                if full_episode_updates:
                    if not er_active and pending_no_er_transitions:
                        for tr_obs, tr_action, tr_reward, tr_next_obs, tr_terminated in pending_no_er_transitions:
                            dqn_update(
                                model, optimizer, tr_obs, tr_action, tr_reward, tr_next_obs,
                                tr_terminated, discount_factor,
                                target_network_step=target_network_step,
                                global_step=global_step,
                                target_network_active=bool(target_network_active),
                            )
                        pending_no_er_transitions = []
                    if er_active and pending_er_updates > 0:
                        for _ in range(pending_er_updates):
                            dqn_batch_update(
                                model, optimizer, replay_buffer, er_batch_size, discount_factor,
                                target_network_step=target_network_step,
                                global_step=global_step,
                                target_network_active=bool(target_network_active),
                            )
                        pending_er_updates = 0
                if pbar is not None:
                    pbar.set_postfix_str(f"episode_reward={total_reward:.2f}", refresh=False)
                last_episode_return = total_reward   # save completed episode return before resetting
                episode += 1
                total_reward = 0.0
            
            if global_step % n_returns_interval == 0:
                if bool(eval_with_env_episode_trials) and eval_env is not None:
                    ep_returns = []
                    for _ in range(int(n_eval_episodes)):
                        obs_e, _info_e = eval_env.reset()
                        ep_ret = 0.0
                        for _ in range(int(max_eval_episode_length)):
                            action = choose_action(
                                model,
                                obs_e,
                                exploration_method="epsilon_greedy",
                                epsilon=0.0,
                            )
                            obs_e, reward_e, terminated_e, truncated_e, _ = eval_env.step(action)
                            ep_ret += float(reward_e)
                            if terminated_e or truncated_e:
                                break
                        ep_returns.append(ep_ret)
                    eval_returns[eval_write_idx] = float(np.mean(ep_returns))
                else:
                    eval_returns[eval_write_idx] = last_episode_return
                eval_time_steps[eval_write_idx] = global_step
                eval_write_idx += 1

        if pbar is not None and total_reward > 0:
            pbar.set_postfix_str(f"episode_reward={total_reward:.2f} (partial)", refresh=False)
            
        if global_step % n_returns_interval != 0:
            if bool(eval_with_env_episode_trials) and eval_env is not None:
                ep_returns = []
                for _ in range(int(n_eval_episodes)):
                    obs_e, _info_e = eval_env.reset()
                    ep_ret = 0.0
                    for _ in range(int(max_eval_episode_length)):
                        action = choose_action(
                            model,
                            obs_e,
                            exploration_method="epsilon_greedy",
                            epsilon=0.0,
                        )
                        obs_e, reward_e, terminated_e, truncated_e, _ = eval_env.step(action)
                        ep_ret += float(reward_e)
                        if terminated_e or truncated_e:
                            break
                    ep_returns.append(ep_ret)
                eval_returns[eval_write_idx] = float(np.mean(ep_returns))
            else:
                eval_returns[eval_write_idx] = last_episode_return
            eval_time_steps[eval_write_idx] = global_step
            eval_write_idx += 1
    finally:
        if shared_step_counter is not None:
            shared_step_counter.value = global_step
        if pbar is not None:
            pbar.close()

        if eval_env is not None:
            eval_env.close()

    model.eval()
    return np.array(eval_returns), np.array(eval_time_steps)  
####################################################################

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, terminated):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, terminated)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)




################[ Main Execution Block             ]################
if __name__ == "__main__":
    preview_animation = show_one_episode(basic_policy)
    plt.show()
####################################################################
