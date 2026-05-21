# Returns summary - mean ± std across all repetitions and eval points (n_repetitions=5)

| Algorithm | Setting | Mean (all) | Std (all) | Mean (last 10%) | Std (last 10%) | N (all) | N (last) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A2C | A2C, TN_step=10, actor_hidden_nn=[64, 64], actor_lr=9.999999747378752e-05, critic_hidden_nn=[128, 128], critic_lr=0.009999999776482582, eval_interval=250, eval_with_env_episode_trials=True, gamma=0.9900000095367432, max_eval_episode_length=500, max_train_episode_length=500, n_eval_episodes=5, n_repetitions=5, n_timesteps=110000 | 499.16 | 4.01 | 498.81 | 4.77 | 2,200 | 220 |
| PPO | PPO, actor_hidden_nn=[128, 128], actor_lr=0.0003000000142492354, clip_eps=0.10000000149011612, critic_hidden_nn=[512, 512], critic_lr=0.009999999776482582, eval_interval=250, eval_with_env_episode_trials=True, gae_lambda=0.9599999785423279, gamma=0.9900000095367432, max_eval_episode_length=500, max_train_episode_length=500, n_epochs=40, n_eval_episodes=5, n_repetitions=5, n_timesteps=110000, rollout_steps=512 | 497.17 | 14.67 | 497.30 | 17.87 | 2,200 | 220 |
