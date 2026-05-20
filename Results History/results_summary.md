# Returns summary - mean ± std across all repetitions and eval points (n_repetitions=5)

| Algorithm | Setting | Mean (all) | Std (all) | Mean (last 10%) | Std (last 10%) | N (all) | N (last) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DQN | DQN, $\gamma$: 0.99, $\alpha$: 0.001, NN Widths: [128,128], $\epsilon$: 0.05, TN: True, ER: True | 411.07 | 139.90 | 420.32 | 131.54 | 20,000 | 2,000 |
| A2C | A2C, $\gamma$: 0.99, Actor $\alpha$: 1e-4, Critic $\beta$: 0.01, Actor NN: [64,64], Critic NN: [128,128] | 500.00 | 0.00 | 500.00 | 0.00 | 20,000 | 2,000 |
| PPO | PPO, $\gamma$: 0.99, Actor $\alpha$: 3e-4, Critic $\beta$: 0.01, $\lambda_{GAE}$: 0.96, $\epsilon_{clip}$: 0.1, Epochs: 30, Rollout: 512 | 498.29 | 15.75 | 498.74 | 17.13 | 20,000 | 2,000 |
