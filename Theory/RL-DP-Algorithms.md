
## Index

**Model-Based Methods**

<details><summary>&emsp;&emsp;&emsp;1. Value Iteration</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-1-1) &nbsp;&middot; [Algorithm](#value-iteration) &nbsp;&middot; [Engineering Tricks](#tricks-1-1) &nbsp;&middot; [Pseudocode](#pseudo-1-1)

</details>

<details><summary>&emsp;&emsp;&emsp;2. Policy Iteration</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-1-2) &nbsp;&middot; [Algorithm](#policy-iteration) &nbsp;&middot; [Engineering Tricks](#tricks-1-2) &nbsp;&middot; [Pseudocode](#pseudo-1-2)

</details>

<details><summary>&emsp;&emsp;&emsp;3. Held-Karp (Bottom-Up / Tabulation)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-1-3) &nbsp;&middot; [Algorithm](#held-karp-bottom-up) &nbsp;&middot; [Engineering Tricks](#tricks-1-3) &nbsp;&middot; [Pseudocode](#pseudo-1-3)

</details>

<details><summary>&emsp;&emsp;&emsp;4. Held-Karp (Top-Down / Memoisation)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-1-4) &nbsp;&middot; [Algorithm](#held-karp-top-down) &nbsp;&middot; [Engineering Tricks](#tricks-1-4) &nbsp;&middot; [Pseudocode](#pseudo-1-4)

</details>

**Model-Free Methods**

&emsp;&emsp;[On-Policy vs Off-Policy Algorithms](#on-off-policy-intro)  
&emsp;&emsp;[Full Episodes vs Step-wise Updates](#full-episodes-intro)

<details><summary>&emsp;&emsp;&emsp;1. Monte Carlo (First-Visit, On-Policy)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-2-1) &nbsp;&middot; [Algorithm](#monte-carlo-first-visit) &nbsp;&middot; [Engineering Tricks](#tricks-2-1) &nbsp;&middot; [Pseudocode](#pseudo-2-1)

</details>

<details><summary>&emsp;&emsp;&emsp;2. Temporal Difference — TD(0)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-2-2) &nbsp;&middot; [Algorithm](#td-0) &nbsp;&middot; [Engineering Tricks](#tricks-2-2) &nbsp;&middot; [Pseudocode](#pseudo-2-2)

</details>

<details><summary>&emsp;&emsp;&emsp;3. n-Step TD</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-2-3) &nbsp;&middot; [Algorithm](#n-step-td) &nbsp;&middot; [Engineering Tricks](#tricks-2-3) &nbsp;&middot; [Pseudocode](#pseudo-2-3)

</details>

<details><summary>&emsp;&emsp;&emsp;4. SARSA (State-Action-Reward-State-Action)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-2-4) &nbsp;&middot; [Algorithm](#sarsa) &nbsp;&middot; [Engineering Tricks](#tricks-2-4) &nbsp;&middot; [Pseudocode](#pseudo-2-4)

</details>

<details><summary>&emsp;&emsp;&emsp;5. Q-Learning</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-2-5) &nbsp;&middot; [Algorithm](#q-learning) &nbsp;&middot; [Engineering Tricks](#tricks-2-5) &nbsp;&middot; [Pseudocode](#pseudo-2-5)

</details>

&emsp;&emsp;**Deep RL — Value-Based**

<details><summary>&emsp;&emsp;&emsp;1. DQN Naive (Deep Q-Network, no tricks)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-3-1) &nbsp;&middot; [Algorithm](#dqn-naive) &nbsp;&middot; [Engineering Tricks](#tricks-3-1) &nbsp;&middot; [Pseudocode](#pseudo-3-1)

</details>

<details><summary>&emsp;&emsp;&emsp;2. DQN + Target Network (DQN+TN)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-3-2) &nbsp;&middot; [Algorithm](#dqn-target-network) &nbsp;&middot; [Engineering Tricks](#tricks-3-2) &nbsp;&middot; [Pseudocode](#pseudo-3-2)

</details>

<details><summary>&emsp;&emsp;&emsp;3. DQN + Experience Replay (DQN+ER)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-3-3) &nbsp;&middot; [Algorithm](#dqn-experience-replay) &nbsp;&middot; [Engineering Tricks](#tricks-3-3) &nbsp;&middot; [Pseudocode](#pseudo-3-3)

</details>

&emsp;&emsp;**Deep RL — Policy Gradient Methods**

&emsp;&emsp;&emsp;&emsp;[What Is the Vanilla Policy Gradient Loss?](#vanilla-pg-loss)

&emsp;&emsp;&emsp;&emsp;<strong>*Vanilla Policy Gradient Methods:*</strong>

<details><summary>&emsp;&emsp;&emsp;1. REINFORCE (Monte Carlo Policy Gradient)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-4-1) &nbsp;&middot; [Algorithm](#reinforce) &nbsp;&middot; [Engineering Tricks](#tricks-4-1) &nbsp;&middot; [Pseudocode](#pseudo-4-1)

</details>

<details><summary>&emsp;&emsp;&emsp;2. Actor-Critic (AC)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-4-2) &nbsp;&middot; [Algorithm](#actor-critic) &nbsp;&middot; [Engineering Tricks](#tricks-4-2) &nbsp;&middot; [Pseudocode](#pseudo-4-2)

</details>

<details><summary>&emsp;&emsp;&emsp;3. A2C (Advantage Actor-Critic)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-4-3) &nbsp;&middot; [Algorithm](#a2c) &nbsp;&middot; [Engineering Tricks](#tricks-4-3) &nbsp;&middot; [Pseudocode](#pseudo-4-3)

</details>

<details><summary>&emsp;&emsp;&emsp;4. A3C (Asynchronous Advantage Actor-Critic)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-4-4) &nbsp;&middot; [Algorithm](#a3c) &nbsp;&middot; [Engineering Tricks](#tricks-4-4) &nbsp;&middot; [Pseudocode](#pseudo-4-4)

</details>

&emsp;&emsp;&emsp;&emsp;<strong>*Advanced Policy Optimization Methods:*</strong>

<details><summary>&emsp;&emsp;&emsp;1. DDPG (Deep Deterministic Policy Gradient)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-4-5) &nbsp;&middot; [Algorithm](#ddpg) &nbsp;&middot; [Engineering Tricks](#tricks-4-5) &nbsp;&middot; [Pseudocode](#pseudo-4-5)

</details>

<details><summary>&emsp;&emsp;&emsp;2. SAC (Soft Actor-Critic)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-4-6) &nbsp;&middot; [Algorithm](#sac) &nbsp;&middot; [Engineering Tricks](#tricks-4-6) &nbsp;&middot; [Pseudocode](#pseudo-4-6)

</details>

<details><summary>&emsp;&emsp;&emsp;3. PPO (Proximal Policy Optimization)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-4-7) &nbsp;&middot; [Algorithm](#ppo) &nbsp;&middot; [Engineering Tricks](#tricks-4-7) &nbsp;&middot; [Pseudocode](#pseudo-4-7)

</details>

<details><summary>&emsp;&emsp;&emsp;4. CQL (Conservative Q-Learning)</summary>

&emsp;&emsp;&emsp;&nbsp;&nbsp;&middot; [Introduction](#intro-4-8) &nbsp;&middot; [Algorithm](#cql) &nbsp;&middot; [Engineering Tricks](#tricks-4-8) &nbsp;&middot; [Pseudocode](#pseudo-4-8)

</details>

<div style="page-break-after: always; break-after: page;"></div>


---
---

<a id='value-iteration'></a>

### Value Iteration:

**Notation:**
- $r(s,a,s')$ — reward received when transitioning from $s$ to $s'$ via action $a$
- $v$ — temporary copy of $V(s)$ before the Bellman update, used to measure change $\Delta$

<a id='intro-1-1'></a>

**Introduction:** Solves an MDP by iteratively applying the Bellman optimality update $V(s) \leftarrow \max_a \sum_{s'} P(s'\mid s,a)\bigl[r(s,a,s') + \gamma  V(s')\bigr]$ until $V$ converges, then reads off the greedy policy. *Idea:* fuse policy evaluation and policy improvement into a single max-backup — no need to fully evaluate each intermediate policy as Policy Iteration does.

**Algorithm:**

**Input:** MDP $(S, A, P, r, \gamma)$, threshold $\theta > 0$

1. Initialize $V(s)$ arbitrarily for all $s \in S$ (e.g., $V(s) = 0$)

2. **Repeat** *(outer loop — sweeps)*:
   - $\Delta \leftarrow 0$
   - For each $s \in S$ *(inner loop — states)*:
     - $v \leftarrow V(s)$
     - $\displaystyle V(s) \leftarrow \max_a \sum_{s'} P(s' \mid s,a) \bigl[r(s,a,s') + \gamma  V(s')\bigr]$
     - $\Delta \leftarrow \max(\Delta,  |v - V(s)|)$
   - Until $\Delta < \theta$ *(convergence check)*

3. **Output policy** — for each $s \in S$:

$$\pi(s) = \arg\max_a \sum_{s'} P(s' \mid s,a) \bigl[r(s,a,s') + \gamma  V(s')\bigr]$$

Return $\pi$

<a id='tricks-1-1'></a>

**Additional Known Engineering Tricks**

- <a id='trick-1-1-1'></a>**In-place (Gauss–Seidel) updates** — overwrite $V(s)$ as soon as the new value is computed within a sweep, rather than building a fresh copy from $V_k$. *Why it helps:* the freshest values are reused immediately by later states in the same sweep, so the effective contraction is tighter and convergence typically needs fewer sweeps. *Interaction:* now the *order* of states matters — pairs naturally with **prioritized sweeping** and **asynchronous DP**, and is mutually exclusive with the *vectorised* synchronous backup below.

- **Prioritized sweeping** *(Moore & Atkeson, 1993)* — maintain a priority queue keyed by each state's current Bellman residual; on every step, pop the highest-error state, update it, and re-queue its predecessors. *Why it helps:* compute is concentrated where $V$ is still changing significantly, instead of wasting full sweeps on already-converged states. *Interaction:* assumes in-place updates; in the limit it is the most aggressive scheduling of asynchronous DP.

- <a id='trick-1-1-2'></a>**Asynchronous DP** — update only a subset of states per pass (random, trajectory-based, or reachable-from-start). *Why it helps:* avoids touching irrelevant states in large MDPs where most states are unreachable from typical start distributions, and parallelises across processors without coordination. *Interaction:* generalises in-place sweeping; convergence still holds as long as every state is visited infinitely often.

- <a id='trick-1-1-3'></a>**Vectorised Bellman backup** — store $P$ as a dense $(|S|,|A|,|S'|)$ tensor and compute one synchronous sweep via `einsum` / matrix products. *Why it helps:* turns the triple loop into a single BLAS call, giving order-of-magnitude wall-clock speed-ups on dense MDPs. *Interaction:* implements the *synchronous* Jacobi-style backup — incompatible with strict in-place updates; pick one or the other depending on whether speed or sweep count matters more.

- <a id='trick-1-1-4'></a>**Span-seminorm convergence test** — replace $\Delta < \theta$ with $\mathrm{sp}(V_{k+1}-V_k) < \theta (1-\gamma)/\gamma$, where $\mathrm{sp}(x)=\max x-\min x$. *Why it helps:* yields a tight $\varepsilon$-optimal-*policy* guarantee from the contraction bound, so the loop stops as soon as the policy is provably near-optimal rather than at an arbitrary max-norm threshold.

- <a id='trick-1-1-5'></a>**Action-set pruning / dominance** — at each state, drop actions that are dominated (lower $Q(s,a)$ than another action under all admissible $V$ bounds). *Why it helps:* shrinks the inner $\max_a$ in every sweep — important when $|A|$ is large or many actions are clearly suboptimal early.

<a id='pseudo-1-1'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 1.1</strong> Value iteration — Raw Pseudocode (without additional tricks) <em>[Sutton & Barto, §4.4]</em>
</div>

Here is the pseudocode for the value iteration core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** MDP $(S, A, P, r, \gamma)$, threshold $\theta > 0$  
Initialize $V(s) \leftarrow 0$ for all $s \in S$  
**repeat**  
&emsp;$\Delta \leftarrow 0$  
&emsp;**for each** $s \in S$ **do**  
&emsp;&emsp;$v \leftarrow V(s)$  
&emsp;&emsp;$V(s) \leftarrow \max_{a}  \sum_{s'} P(s' \mid s, a) \bigl[  r(s, a, s') + \gamma  V(s')  \bigr]$  
&emsp;&emsp;$\Delta \leftarrow \max\bigl(\Delta,  |v - V(s)|\bigr)$  
&emsp;**end for**  
**until** $\Delta < \theta$  
**Output:** deterministic policy $\pi(s) \leftarrow \arg\max_{a}  \sum_{s'} P(s' \mid s, a) \bigl[  r(s, a, s') + \gamma  V(s')  \bigr]$  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 1.1</strong> Value iteration — Pseudocode with Additions (Engineering Tricks) <em>[Sutton & Barto, §4.4]</em>
</div>

Here is the pseudocode for the value iteration algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** MDP $(S, A, P, r, \gamma)$, threshold $\theta > 0$  
Initialize $V(s) \leftarrow 0$ for all $s \in S$  
✦ Initialize priority queue $\mathrm{PQ}$; insert every $s$ with priority $\infty$ ✦ ([Prioritized sweeping](#trick-1-1-2))  
✦ Pre-compute predecessor sets $\mathrm{pred}(s)$ for all $s$ ✦ ([Prioritized sweeping](#trick-1-1-2))  
**repeat**  
&emsp;$\Delta \leftarrow 0$  
&emsp;✦ **while** $\mathrm{PQ}$ is not empty **do** *(update only high-residual states; or process a random / trajectory-based subset)* ✦ ([Prioritized sweeping](#trick-1-1-2) / [Asynchronous DP](#trick-1-1-3))  
&emsp;&emsp;✦ $s \leftarrow \mathrm{PQ.popMax}()$ ✦ ([Prioritized sweeping](#trick-1-1-2))  
&emsp;&emsp;$v \leftarrow V(s)$  
&emsp;&emsp;✦ $A'(s) \leftarrow \lbrace a \in A \mid a \text{ not dominated}\rbrace$ ✦ ([Action-set pruning / dominance](#trick-1-1-6))  
&emsp;&emsp;✦ $V(s) \leftarrow \max_{a \in A'(s)}  \sum_{s'} P(s' \mid s, a) \bigl[  r(s, a, s') + \gamma  V(s')  \bigr]$ *(overwrite in-place)* ✦ ([In-place (Gauss–Seidel) updates](#trick-1-1-1))  
&emsp;&emsp;$\Delta \leftarrow \max\bigl(\Delta,  |v - V(s)|\bigr)$  
&emsp;&emsp;✦ **for each** $s_p \in \mathrm{pred}(s)$ **do** update $\mathrm{PQ}(s_p)$ with new residual **end for** ✦ ([Prioritized sweeping](#trick-1-1-2))  
&emsp;✦ **end while** ✦ ([Prioritized sweeping](#trick-1-1-2))  
✦ **until** $\mathrm{sp}(V_{k+1} - V_k) < \theta (1-\gamma)/\gamma$ *(span-seminorm test)* ✦ ([Span-seminorm convergence test](#trick-1-1-5))  
✦ *Alternative: replace the per-state loop with $V \leftarrow$ vectorised $\max_a\bigl[P_a (r_a + \gamma V)\bigr]$ via einsum (synchronous Jacobi-style; incompatible with in-place)* ✦ ([Vectorised Bellman backup](#trick-1-1-4))  
**Output:** $\pi(s) \leftarrow \arg\max_{a \in A'(s)}  \sum_{s'} P(s' \mid s, a) \bigl[  r(s, a, s') + \gamma  V(s')  \bigr]$  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='policy-iteration'></a>

### Policy Iteration

**Notation:**
- $r(s,a,s')$ — reward on transition $s \xrightarrow{a} s'$
- $v$ — temporary copy of $V(s)$ before update
- $\text{policyStable}$ — flag that becomes false whenever any state's action changes during improvement

<a id='intro-1-2'></a>

**Introduction:** Solves an MDP by alternating two steps until the policy stops changing: (i) **policy evaluation** — compute $V^\pi$ for the current $\pi$ by repeated Bellman expectation updates, and (ii) **policy improvement** — set $\pi(s) \leftarrow \arg\max_a \sum_{s'} P(s'\mid s,a)\bigl[r + \gamma  V^\pi(s')\bigr]$. *Idea:* each improvement is guaranteed to produce a strictly better (or equal) policy (policy improvement theorem), so the algorithm converges in *finitely* many outer iterations on a finite MDP — typically far fewer than Value Iteration needs sweeps, at the cost of a full evaluation sweep per outer iteration.

**Algorithm:**

**Input:** MDP $(S, A, P, r, \gamma)$, threshold $\theta > 0$

1. Initialize $\pi(s)$ arbitrarily for all $s \in S$

2. **Repeat** *(outer loop — policy iterations)*:

   - **— Policy Evaluation —**

   - **Repeat** *(middle loop — evaluation sweeps)*:
     - $\Delta \leftarrow 0$
     - For each $s \in S$ *(inner loop — states)*:
       - $v \leftarrow V(s)$
       - $\displaystyle V(s) \leftarrow \sum_{s'} P\left(s' \mid s,\pi(s)\right)\bigl[r(s,\pi(s),s') + \gamma  V(s')\bigr]$
       - $\Delta \leftarrow \max(\Delta,  |v - V(s)|)$
     - Until $\Delta < \theta$ *(evaluation convergence check)*

   - **— Policy Improvement —**

   - $\text{policyStable} \leftarrow \textit{true}$
   - For each $s \in S$ *(single pass — states)*:
     - $\text{oldAction} \leftarrow \pi(s)$
     - $\displaystyle \pi(s) \leftarrow \arg\max_a \sum_{s'} P(s' \mid s,a) \bigl[r(s,a,s') + \gamma  V(s')\bigr]$
     - If $\text{oldAction} \neq \pi(s)$: $\text{policyStable} \leftarrow \textit{false}$

   Until $\text{policyStable} = \textit{true}$ *(outer convergence check)*

3. Return $\pi,  V$

<a id='tricks-1-2'></a>

**Additional Known Engineering Tricks**

- **Modified Policy Iteration (MPI)** *(Puterman & Shin, 1978)* — replace exact policy evaluation with just $k$ Bellman-expectation sweeps (typically $k\in[5,50]$). *Why it helps:* full evaluation is wasted effort early on when the policy is still bad and will change anyway — partial evaluation is enough to drive improvement, so total work drops dramatically. *Interaction:* defines the spectrum from value iteration ($k=1$) to classical policy iteration ($k=\infty$).

- <a id='trick-1-2-1'></a>**Generalised Policy Iteration (GPI)** — interleave evaluation and improvement at *any* granularity (one state at a time, asynchronously, even action-selection-driven). *Why it helps:* makes the algorithm an anytime procedure that converges as long as both processes make progress on every state infinitely often. *Interaction:* MPI is the special case where each interleaved evaluation phase is $k$ full sweeps; pairs with **asynchronous policy evaluation**.

- <a id='trick-1-2-2'></a>**Warm-start $V$ across outer iterations** — keep $V$ between policy improvements rather than re-initialising to zero. *Why it helps:* a newly-improved policy's value function is usually close to the previous one, so inner evaluation needs only a few sweeps to re-converge. *Interaction:* this is precisely what makes MPI's small $k$ work — without warm-start, truncating evaluation would forget all prior progress.

- <a id='trick-1-2-3'></a>**Action-value caching during improvement** — during the improvement sweep, compute $Q(s,a)=\sum_{s'} P(s'\mid s,a)[r+\gamma V(s')]$ once per $(s,a)$, store it, then take $\arg\max$ and the change-flag from the same table. *Why it helps:* the improvement step otherwise recomputes the same sum twice — once for the argmax, once when checking if anything changed.

- <a id='trick-1-2-4'></a>**Policy-stable early exit** — break the inner evaluation as soon as no state's greedy action would change. *Why it helps:* once the policy has stabilised, further evaluation cannot improve it; the outer loop will terminate next round anyway.

- <a id='trick-1-2-5'></a>**Bertsekas's $\lambda$-policy-iteration** — bootstrap evaluation with $\lambda$-returns instead of one-step Bellman backups. *Why it helps:* trades a controllable amount of bias for much faster propagation of value information across long horizons. *Interaction:* generalises both MPI and value iteration as endpoints.

<a id='pseudo-1-2'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 1.2</strong> Policy iteration — Raw Pseudocode (without additional tricks) <em>[Sutton & Barto, §4.3]</em>
</div>

Here is the pseudocode for the policy iteration core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** MDP $(S, A, P, r, \gamma)$, threshold $\theta > 0$  
Initialize $V(s) \in \mathbb{R}$ and $\pi(s) \in A$ arbitrarily for all $s \in S$  
**repeat**  
&emsp;*// — Policy Evaluation —*  
&emsp;**repeat**  
&emsp;&emsp;$\Delta \leftarrow 0$  
&emsp;&emsp;**for each** $s \in S$ **do**  
&emsp;&emsp;&emsp;$v \leftarrow V(s)$  
&emsp;&emsp;&emsp;$V(s) \leftarrow \sum_{s'} P(s' \mid s, \pi(s)) \bigl[  r(s, \pi(s), s') + \gamma  V(s')  \bigr]$  
&emsp;&emsp;&emsp;$\Delta \leftarrow \max\bigl(\Delta,  |v - V(s)|\bigr)$  
&emsp;&emsp;**end for**  
&emsp;**until** $\Delta < \theta$  
&emsp;*// — Policy Improvement —*  
&emsp;$\text{policyStable} \leftarrow \mathbf{true}$  
&emsp;**for each** $s \in S$ **do**  
&emsp;&emsp;$a_{\text{old}} \leftarrow \pi(s)$  
&emsp;&emsp;$\pi(s) \leftarrow \arg\max_{a}  \sum_{s'} P(s' \mid s, a) \bigl[  r(s, a, s') + \gamma  V(s')  \bigr]$  
&emsp;&emsp;**if** $a_{\text{old}} \neq \pi(s)$ **then** $\text{policyStable} \leftarrow \mathbf{false}$  
&emsp;**end for**  
**until** $\text{policyStable} = \mathbf{true}$  
**Output:** $V \approx v_{*}$ and $\pi \approx \pi_{*}$  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 1.2</strong> Policy iteration — Pseudocode with Additions (Engineering Tricks) <em>[Sutton & Barto, §4.3]</em>
</div>

Here is the pseudocode for the policy iteration algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** MDP $(S, A, P, r, \gamma)$, threshold $\theta > 0$, ✦ evaluation sweeps $k$ ✦ ([Modified Policy Iteration (MPI)](#trick-1-2-1))  
Initialize $V(s) \in \mathbb{R}$ and $\pi(s) \in A$ arbitrarily for all $s \in S$  
**repeat**  
&emsp;*// — Policy Evaluation (truncated) —*  
&emsp;✦ **for** $\text{sweep} = 1, \ldots, k$ **do** *(only $k$ sweeps instead of until convergence)* ✦ ([Modified Policy Iteration (MPI)](#trick-1-2-1))  
&emsp;&emsp;**for each** $s \in S$ **do**  
&emsp;&emsp;&emsp;$V(s) \leftarrow \sum_{s'} P(s' \mid s, \pi(s)) \bigl[  r(s, \pi(s), s') + \gamma  V(s')  \bigr]$  
&emsp;&emsp;**end for**  
&emsp;&emsp;✦ **if** no state's greedy action would change **then break** *(early exit)* ✦ ([Policy-stable early exit](#trick-1-2-5))  
&emsp;✦ **end for** *(warm-start: $V$ is carried forward across outer iterations, not re-initialised)* ✦ ([Warm-start $V$ across outer iterations](#trick-1-2-3))  
&emsp;*// — Policy Improvement —*  
&emsp;$\text{policyStable} \leftarrow \mathbf{true}$  
&emsp;**for each** $s \in S$ **do**  
&emsp;&emsp;$a_{\text{old}} \leftarrow \pi(s)$  
&emsp;&emsp;✦ Cache $Q(s,a) \leftarrow \sum_{s'} P(s'\mid s,a)\bigl[r + \gamma V(s')\bigr]$ for all $a$ *(compute once, reuse for argmax and stability check)* ✦ ([Action-value caching during improvement](#trick-1-2-4))  
&emsp;&emsp;✦ $\pi(s) \leftarrow \arg\max_a Q(s,a)$ ✦ ([Action-value caching during improvement](#trick-1-2-4))  
&emsp;&emsp;**if** $a_{\text{old}} \neq \pi(s)$ **then** $\text{policyStable} \leftarrow \mathbf{false}$  
&emsp;**end for**  
&emsp;✦ *(Evaluation and improvement may be interleaved at any granularity — per-state, per-action — under the GPI framework)* ✦ ([Generalised Policy Iteration (GPI)](#trick-1-2-2))  
**until** $\text{policyStable} = \mathbf{true}$  
**Output:** $V \approx v_{*}$ and $\pi \approx \pi_{*}$  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='held-karp-bottom-up'></a>

### Held-Karp (Bottom-Up / Tabulation)

**Notation:**
- $n$ — total number of cities
- $d[i][j]$ — travel cost / distance from city $i$ to city $j$
- $S$ — a *subset* of cities $\lbrace 2,\ldots,n\rbrace$ (not the MDP state space; city $1$ is the fixed start)
- $C(S, j)$ — minimum-cost path: starts at city $1$, visits every city in $S$ exactly once, ends at $j \in S$
- $\text{OPT}$ — optimal (minimum-cost) Hamiltonian tour

<a id='intro-1-3'></a>

**Introduction:** Exact DP algorithm for the Traveling Salesman Problem (TSP). Solves in $O(n^2  2^n)$ time and $O(n  2^n)$ space — exponential, but far better than the naive $O(n!)$ brute-force. *Bottom-up variant:* fills the DP table iteratively from smallest subsets to largest.

**Problem:** Given $n$ cities $\lbrace 1, 2, \ldots, n\rbrace$ and a distance matrix $d[i][j]$, find the shortest Hamiltonian cycle (visit every city exactly once and return to the start). Fix city $1$ as the starting city without loss of generality.

**State definition:** $C(S, j)$ = minimum cost of a path that starts at city $1$, visits every city in subset $S \subseteq \lbrace 2, \ldots, n\rbrace$, and ends at city $j \in S$.

**Algorithm:**

**Input:** number of cities $n$, travel cost matrix $d[i][j]$

1. **Base cases** — paths of length 1 from city $1$ to each other city:

   - For each $l \in \lbrace 2, \ldots, n\rbrace$:
     $$C(\lbrace l\rbrace ,  l) = d[1][l]$$

2. **Fill table bottom-up** — iterate over subsets of increasing size:

   - **For** $|S| = 2, 3, \ldots, n{-}1$ *(outer loop — subset sizes)*:
     - **For each** subset $S \subseteq \lbrace 2, \ldots, n\rbrace$ with $|S|$ elements *(middle loop — subsets)*:
       - **For each** $l \in S$ *(inner loop — ending city)*:
         $$C(S,  l) = \min_{k \in S \setminus \lbrace l\rbrace } \bigl[C(S \setminus \lbrace l\rbrace ,  k) + d[k][l]\bigr]$$
         *"Best way to reach $l$ = best way to reach some predecessor $k$ in $S \setminus \lbrace l\rbrace$, plus edge $k \to l$"*

3. **Close the tour** — return to city $1$:

$$\text{OPT} = \min_{j \in \lbrace 2,\ldots,n\rbrace } \bigl[C(\lbrace 2,\ldots,n\rbrace ,  j) + d[j][1]\bigr]$$

4. **Reconstruct path** by backtracking through stored argmin choices.

   Return optimal tour and cost $\text{OPT}$



**Notes:**

- **Bitmask encoding:** subsets $S$ are represented as bitmasks of $n{-}1$ bits, e.g. $S = \lbrace 2,4\rbrace$ with $n=5$ is encoded as $\texttt{0101}_2 = 5$. This allows $O(1)$ set operations (add, remove, membership) via bitwise operators.
- **Time complexity:** $O(n^2  2^n)$ — for each of $2^{n-1}$ subsets, we iterate over $O(n)$ ending cities and $O(n)$ predecessors.
- **Space complexity:** $O(n  2^n)$ — the DP table $C[S][j]$ stores one value per (subset, ending city) pair.
- Practical limit: $n \approx 20\text{–}25$ cities (due to exponential memory/time).
- The bottom-up approach fills the entire table even if some entries are never needed for the final answer.

<a id='tricks-1-3'></a>

**Additional Known Engineering Tricks**

- <a id='trick-1-3-1'></a>**Bitmask subset encoding** — represent the visited set $S$ as an $(n{-}1)$-bit integer; add/remove a city becomes `mask | (1<<i)` / `mask & ~(1<<i)`. *Why it helps:* set operations are $O(1)$ machine instructions instead of $O(n)$ container manipulations; the DP table can be a flat array indexed by `(mask, j)`. *Interaction:* prerequisite for every other low-level trick below — the pseudocode uses abstract set notation $S\subseteq\lbrace 2,\ldots,n\rbrace$; the bitmask is the standard implementation realisation.

- <a id='trick-1-3-2'></a>**Gosper's hack for fixed-cardinality enumeration** — given a mask with $k$ bits, the next mask with $k$ bits is `t = mask | (mask-1); next = (t+1) | (((~t & -~t)-1) >> (__ctz(mask)+1))`. *Why it helps:* iterates over all $\binom{n-1}{k}$ subsets of size $k$ in constant time per step, instead of skipping over the other $2^{n-1}-\binom{n-1}{k}$ masks.

- <a id='trick-1-3-3'></a>**Flat 2D table $C[\text{mask} \cdot n + j]$** — store the DP values in a contiguous $2^{n-1}\cdot n$ array instead of a nested dict/2D-array-of-arrays. *Why it helps:* cache-friendly linear access pattern in the inner $\min_k$ loop; on modern hardware this is the single biggest constant-factor win.

- <a id='trick-1-3-4'></a>**Parent / predecessor table** — alongside $C[S][j]$, store the argmin $k$ that achieved it. *Why it helps:* enables $O(n)$ path reconstruction by walking back through the parent pointers, avoiding a second DP pass.

- <a id='trick-1-3-5'></a>**Symmetry exploitation for symmetric distance matrices** — when $d[i][j]=d[j][i]$, the tour and its reverse have equal cost, so you may fix not just the start city but also the *direction* (e.g., require the second city to have the smallest index among $S\setminus\lbrace 1\rbrace$). *Why it helps:* halves the state space and runtime constant factor.

- <a id='trick-1-3-6'></a>**Best-known-tour pruning** — compute a quick heuristic upper bound (nearest-neighbour or Christofides) before the DP, then skip any partial $C(S,j)+d[j][1]$ that already exceeds it. *Why it helps:* turns Held–Karp into a branch-and-bound variant that prunes large swaths of the table on structured instances.

<a id='pseudo-1-3'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 1.3</strong> Held–Karp (bottom-up / tabulation) — Raw Pseudocode (without additional tricks) <em>[Held & Karp, 1962]</em>
</div>

Here is the pseudocode for the Held–Karp (bottom-up / tabulation) core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** number of cities $n$, distance matrix $d[i][j]$  
*// Base cases — singleton subsets*  
**for each** $l \in \lbrace 2, \ldots, n\rbrace$ **do**  
&emsp;$C(\lbrace l\rbrace ,  l) \leftarrow d[1][l]$  
**end for**  
*// Build subsets bottom-up by size*  
**for** $s = 2$ **to** $n - 1$ **do**  
&emsp;**for each** $S \subseteq \lbrace 2, \ldots, n\rbrace$ with $|S| = s$ **do**  
&emsp;&emsp;**for each** $l \in S$ **do**  
&emsp;&emsp;&emsp;$C(S,  l) \leftarrow \displaystyle\min_{k \in S \setminus \lbrace l\rbrace } \bigl\lbrace   C(S \setminus \lbrace l\rbrace ,  k) + d[k][l]  \bigr\rbrace$  
&emsp;&emsp;**end for**  
&emsp;**end for**  
**end for**  
*// Close the tour back to city 1*  
$\text{OPT} \leftarrow \displaystyle\min_{l \in \lbrace 2, \ldots, n\rbrace } \bigl\lbrace   C(\lbrace 2, \ldots, n\rbrace ,  l) + d[l][1]  \bigr\rbrace$  
**Output:** $\text{OPT}$  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 1.3</strong> Held–Karp (bottom-up / tabulation) — Pseudocode with Additions (Engineering Tricks) <em>[Held & Karp, 1962]</em>
</div>

Here is the pseudocode for the Held–Karp (bottom-up / tabulation) algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** number of cities $n$, distance matrix $d[i][j]$  
✦ Represent every subset $S\subseteq\lbrace 2,\ldots,n\rbrace$ as an $(n{-}1)$-bit integer ✦ ([Bitmask subset encoding](#trick-1-3-1))  
✦ Allocate flat table $C[\text{mask}\cdot n + j]$ of size $2^{n-1}\cdot n$ ✦ ([Flat 2D table](#trick-1-3-3))  
✦ Allocate parent table $P[\text{mask}\cdot n + j]$ for tour reconstruction ✦ ([Parent / predecessor table](#trick-1-3-4))  
✦ Compute heuristic upper bound $\mathrm{UB}$ (e.g., nearest-neighbour tour) ✦ ([Best-known-tour pruning](#trick-1-3-6))  
*// Base cases — singleton subsets*  
**for each** $l \in \lbrace 2, \ldots, n\rbrace$ **do**  
&emsp;$C[\texttt{1<<}(l{-}2),  l] \leftarrow d[1][l]$  
&emsp;✦ $P[\texttt{1<<}(l{-}2),  l] \leftarrow 1$ ✦ ([Parent / predecessor table](#trick-1-3-4))  
**end for**  
*// Build subsets bottom-up by size*  
**for** $s = 2$ **to** $n - 1$ **do**  
&emsp;✦ **for each** mask with $\mathrm{popcount}(\text{mask}) = s$ *(via Gosper's hack)* **do** ✦ ([Gosper's hack for fixed-cardinality enumeration](#trick-1-3-2))  
&emsp;&emsp;✦ *(For symmetric $d$: fix direction so second city $<$ last city in subset)* ✦ ([Symmetry exploitation for symmetric distance matrices](#trick-1-3-5))  
&emsp;&emsp;**for each** $l$ with bit $l$ set in mask **do**  
&emsp;&emsp;&emsp;$C[\text{mask},  l] \leftarrow \min_{k \in \text{mask}\setminus\lbrace l\rbrace } \bigl\lbrace   C[\text{mask}\oplus\texttt{1<<}(l{-}2),  k] + d[k][l]  \bigr\rbrace$  
&emsp;&emsp;&emsp;✦ $P[\text{mask},  l] \leftarrow \arg\min_k(\cdots)$ ✦ ([Parent / predecessor table](#trick-1-3-4))  
&emsp;&emsp;&emsp;✦ **if** $C[\text{mask},l] + d[l][1] \geq \mathrm{UB}$ **then skip** ✦ ([Best-known-tour pruning](#trick-1-3-6))  
&emsp;&emsp;**end for**  
&emsp;**end for**  
**end for**  
$\text{OPT} \leftarrow \min_{l\in\lbrace 2,\ldots,n\rbrace } \bigl\lbrace   C[\text{full}, l] + d[l][1]  \bigr\rbrace$  
✦ Reconstruct optimal tour by following $P[\cdot]$ backwards from $\arg\min$ ✦ ([Parent / predecessor table](#trick-1-3-4))  
**Output:** $\text{OPT}$ and optimal tour  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='held-karp-top-down'></a>

### Held-Karp (Top-Down / Memoisation)

**Notation:**

Same as in the Bottom-Up variant above:
- $n$ — total number of cities
- $d[i][j]$ — travel cost / distance from city $i$ to city $j$
- $S$ — a *subset* of cities $\lbrace 2,\ldots,n\rbrace$ (not the MDP state space; city $1$ is the fixed start)
- $C(S, j)$ — minimum-cost path: starts at city $1$, visits every city in $S$ exactly once, ends at $j \in S$
- $\text{OPT}$ — optimal (minimum-cost) Hamiltonian tour

- $\texttt{solve}(S, j)$ — recursive function that computes $C(S,j)$; returns the cached value immediately if already computed
- $\texttt{None}$ — sentinel value indicating a subproblem has not yet been evaluated

<a id='intro-1-4'></a>

**Introduction:** Same DP recurrence as the bottom-up variant, but computed ***recursively with memoisation*** — only subproblems actually needed are evaluated.

**Problem:** Same as above — shortest Hamiltonian cycle over $n$ cities with distance matrix $d[i][j]$, starting and ending at city $0$.

**State definition:** $C(S, j)$ = minimum cost of a path starting at city $0$, visiting every city in $S$, ending at $j \in S$.

**Algorithm:**

**Input:** number of cities $n$, distance matrix $d[i][j]$

1. Initialise memo table: $C[S][j] \leftarrow \texttt{None}$ for all $(S, j)$

2. Define recursive function $\texttt{solve}(S, j)$:

   - **Base case:** if $S = \lbrace j\rbrace$, return $d[0][j]$
   - **Memo hit:** if $C[S][j] \neq \texttt{None}$, return $C[S][j]$
   - **Recursive case:**
     $$C(S,  j) = \min_{k \in S \setminus \lbrace j\rbrace } \bigl[\texttt{solve}(S \setminus \lbrace j\rbrace ,  k) + d[k][j]\bigr]$$
   - Store result in $C[S][j]$ and return it

3. **Close the tour:**

$$\text{OPT} = \min_{j \in \lbrace 1,\ldots,n-1\rbrace } \bigl[\texttt{solve}(\lbrace 1,\ldots,n{-}1\rbrace ,  j) + d[j][0]\bigr]$$

4. **Reconstruct path** by backtracking through stored argmin choices.

   Return optimal tour and cost $\text{OPT}$

✦ ✦ ✦

**Notes:**

- **Same asymptotic complexity** as bottom-up: $O(n^2  2^n)$ time, $O(n  2^n)$ space (worst case all subproblems are visited).
- **Advantage over bottom-up:** only computes subproblems that are actually reachable from the final answer — can be faster in practice if large parts of the state space are unreachable.
- **Disadvantage:** recursive call overhead and potential stack depth issues for large $n$. Python's default recursion limit may need to be increased.
- **Bitmask encoding** is used identically: $S$ is an integer bitmask, memo table is a dictionary or 2D array indexed by $(\text{bitmask}, j)$.
- **Comparison:**

| | Bottom-Up (Tabulation) | Top-Down (Memoisation) |
|---|---|---|
| Approach | Iterative; fills entire table | Recursive; fills on demand |
| Subproblems computed | All $O(n  2^n)$ entries | Only reachable entries |
| Overhead | Loop overhead only | Recursion + memo lookup |
| Implementation | Nested loops over subset sizes | Recursive function + cache |
| Memory pattern | Sequential, cache-friendly | Scattered, less cache-friendly |

<a id='tricks-1-4'></a>

**Additional Known Engineering Tricks**

- <a id='trick-1-4-1'></a>**`functools.lru_cache` / explicit hash memo** — wrap `solve(S, j)` with `@lru_cache(maxsize=None)` (or use a manual `dict` keyed by `(mask, j)`). *Why it helps:* one line gives you the memoisation that defines top-down DP, with proper hashing and eviction-free behaviour. *Interaction:* requires hashable keys — use `int` bitmasks (next bullet).

- <a id='trick-1-4-2'></a>**Integer bitmask keys instead of `frozenset`** — represent $S$ as a Python `int` rather than a `frozenset`/`tuple`. *Why it helps:* `int` hashing and equality are far cheaper than for frozensets; lookup speed dominates Held–Karp top-down runtime.

- <a id='trick-1-4-3'></a>**Recursion-limit bump and stack-safe iteration** — call `sys.setrecursionlimit(10**6)` for $n\geq20$, or convert `solve` to an iterative form with an explicit `deque`-based stack. *Why it helps:* default CPython recursion limit (1000) is exceeded around $n\approx 18$ in the worst case; without the bump the algorithm crashes silently as $n$ grows.

- <a id='trick-1-4-4'></a>**Branch-and-bound with global upper bound** — keep a running best total-tour cost; inside `solve(S, j)`, if the partial cost plus a *lower bound* on the remaining $C(\bar S, \cdot)$ (e.g., sum of $\min$ outgoing edges) exceeds the best, return $+\infty$. *Why it helps:* turns memoisation into pruning — many subproblems are never expanded. *Interaction:* only helps when a good initial heuristic tour exists (e.g., nearest-neighbour upper bound).

- <a id='trick-1-4-5'></a>**Lazy memo (dict not array)** — store only the subproblems actually visited, not the full $O(n\cdot 2^n)$ table. *Why it helps:* on instances where large parts of the state space are unreachable, memory drops by orders of magnitude — the headline advantage over bottom-up tabulation.

- <a id='trick-1-4-6'></a>**Tail-shared subproblem detection** — observe that `solve(S, j)` and `solve(S, j')` share all `solve(S\lbrace j}, k)` sub-calls; batch them in a single sweep over $S$ at the end of each subproblem. *Why it helps:* reduces redundant memo lookups; mirrors what bottom-up gets for free.

<a id='pseudo-1-4'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 1.4</strong> Held–Karp (top-down / memoisation) — Raw Pseudocode (without additional tricks) <em>[Held & Karp, 1962]</em>
</div>

Here is the pseudocode for the Held–Karp (top-down / memoisation) core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** number of cities $n$, distance matrix $d[i][j]$  
Initialize memo table $C[S][j] \leftarrow \texttt{None}$ for all $(S, j)$  

**function** $\texttt{solve}(S,  j)$:  
&emsp;**if** $S = \lbrace j\rbrace$ **then return** $d[1][j]$  
&emsp;**if** $C[S][j] \neq \texttt{None}$ **then return** $C[S][j]$  
&emsp;$C[S][j] \leftarrow \displaystyle\min_{k \in S \setminus \lbrace j\rbrace } \bigl\lbrace   \texttt{solve}(S \setminus \lbrace j\rbrace ,  k) + d[k][j]  \bigr\rbrace$  
&emsp;**return** $C[S][j]$  

$\text{OPT} \leftarrow \displaystyle\min_{j \in \lbrace 2, \ldots, n\rbrace } \bigl\lbrace   \texttt{solve}(\lbrace 2, \ldots, n\rbrace ,  j) + d[j][1]  \bigr\rbrace$  
**Output:** $\text{OPT}$  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 1.4</strong> Held–Karp (top-down / memoisation) — Pseudocode with Additions (Engineering Tricks) <em>[Held & Karp, 1962]</em>
</div>

Here is the pseudocode for the Held–Karp (top-down / memoisation) algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** number of cities $n$, distance matrix $d[i][j]$  
✦ $\texttt{sys.setrecursionlimit}(10^6)$ *(or convert to iterative form with explicit stack)* ✦ ([Recursion-limit bump and stack-safe iteration](#trick-1-4-3))  
✦ Initialise memo as empty $\texttt{dict}$: $C = \lbrace \rbrace$ *(lazy — only visited subproblems stored)* ✦ ([Lazy memo (dict not array)](#trick-1-4-5))  
✦ Compute heuristic upper bound $\mathrm{UB}$ (nearest-neighbour tour) ✦ ([Branch-and-bound with global upper bound](#trick-1-4-4))  

✦ `@lru_cache(maxsize=None)` ✦ ([functools.lru\_cache / explicit hash memo](#trick-1-4-1))  
**function** $\texttt{solve}($✦ $\text{mask}$ ✦ ([Integer bitmask keys instead of frozenset](#trick-1-4-2))$,  j)$:  
&emsp;**if** $\text{mask} = \texttt{1<<}(j{-}2)$ **then return** $d[1][j]$  
&emsp;✦ $\texttt{lb} \leftarrow$ lower bound on remaining tour from $(\text{mask}, j)$ ✦ ([Branch-and-bound with global upper bound](#trick-1-4-4))  
&emsp;✦ **if** partial cost $+ \texttt{lb} \geq \mathrm{UB}$ **then return** $+\infty$ ✦ ([Branch-and-bound with global upper bound](#trick-1-4-4))  
&emsp;✦ Batch all $k\in\text{mask}\setminus\lbrace j\rbrace$ sub-calls sharing the same sub-mask ✦ ([Tail-shared subproblem detection](#trick-1-4-6))  
&emsp;$C[\text{mask}][j] \leftarrow \min_{k \in \text{mask}\setminus\lbrace j\rbrace } \bigl\lbrace   \texttt{solve}(\text{mask}\oplus\texttt{1<<}(j{-}2),  k) + d[k][j]  \bigr\rbrace$  
&emsp;**return** $C[\text{mask}][j]$  

$\text{OPT} \leftarrow \min_{j\in\lbrace 2,\ldots,n\rbrace } \bigl\lbrace   \texttt{solve}(\texttt{fullMask},  j) + d[j][1]  \bigr\rbrace$  
**Output:** $\text{OPT}$  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>


✦ ✦ ✦
## Model-Free Methods
✦ ✦ ✦

<a id='on-off-policy-intro'></a>

### On-Policy vs Off-Policy Algorithms

Model-free control methods learn $Q$ from sampled transitions, but they differ in *which* policy generates those samples vs. *which* policy is being improved.

- **Behavior policy** $\mu$ — the policy actually used to act in the environment (generates the data).
- **Target policy** $\pi$ — the policy whose value we are learning / trying to improve.

$\cdot$ **On-policy:** $\mu = \pi$. The agent learns about the same policy it follows, including its exploration. *Examples: MC control, SARSA, REINFORCE.*

$\cdot$ **Off-policy:** $\mu \neq \pi$. **The agent follows an exploratory behavior policy $\mu$** but learns about a different (often greedy) target policy $\pi$. *Examples: Q-learning, DQN.*

| | On-Policy | Off-Policy |
|---|---|---|
| Data source | Current policy only | Any policy (incl. old / human / replay) |
| TD target uses | Action sampled from $\pi$ (e.g. $Q(s',a')$) | Action chosen by target $\pi$ (e.g. $\max_{a'} Q(s',a')$) |
| Sample efficiency | Lower — old data becomes stale | Higher — can reuse past experience (replay) |
| Exploration cost | Reflected in learned values (safer) | Ignored in target (can be riskier) |
| Stability | Generally more stable | Can diverge with function approximation |

#### More elaborate detail about off-policy vs on-policy

Although the definitions are told above, later you might be confused by some upcoming materials. Here is a more elaborate disambiguation:

- **Definition 1 (strict / Sutton & Barto)**
On-policy: the policy generating data = the policy being evaluated/improved.
Off-policy: they differ.
Under this strict definition, yes — PPO with K>1 epochs is technically off-policy starting at epoch 1, because π_θ ≠ π_old. And the importance ratio π_θ/π_old is literally an importance sampling correction, which is the textbook off-policy tool.
By this definition, A2C with K=1 is on-policy (one update, then throw data away), but A2C with K>1 would also become technically off-policy — just like PPO.
- **Definition 2 (practical / common usage)**
On-policy: uses only recently collected data, discards it after a few updates, relies on importance sampling at most for nearby distributions.
Off-policy: uses a replay buffer with arbitrarily old data, learns a value function that doesn't depend on the behavior policy (Q-learning).
Under this practical definition, PPO and A2C are on-policy because the data is essentially fresh and the policies are kept close. DQN and SAC are off-policy because they reuse data from policies arbitrarily far in the past.

Under this practical definition, PPO and A2C are on-policy because the data is essentially fresh and the policies are kept close. DQN and SAC are off-policy because they reuse data from policies arbitrarily far in the past.

#### Summary Off-policy/On-policy table:

|Algorithm|Off-policy|On-policy|Reason|
|---|---|---|---|
|Monte Carlo control|✔|✔|Standard MC control learns from episodes of the policy being improved; an off-policy variant exists via importance sampling.<br>**Mostly known for:** On-policy|
|SARSA|✘|✔|TD target uses the next action actually taken by the current policy ($Q(s',a')$).|
|Expected SARSA|✔|✔|Target averages $Q(s',\cdot)$ over the target policy; on-policy when target = behavior, off-policy when they differ (greedy target ⇒ Q-learning).<br>**Mostly known for:** On-policy|
|$n$-step SARSA|✘|✔|Multi-step return bootstrapped from actions taken by the current policy.|
|Q-learning|✔|✘|Target uses $\max_{a'} Q(s',a')$ (greedy $\pi$), independent of the exploratory behavior policy $\mu$.|
|DQN|✔|✘|Q-learning with a replay buffer — learns the greedy policy from arbitrarily old transitions.|
|REINFORCE|✘|✔|Policy gradient estimated from trajectories sampled by the current policy.|
|A2C / A3C|✘|✔|Actor-critic updates computed from fresh rollouts of the current policy.|
|TRPO|✔|✔|Uses importance sampling within a trust region, so technically off-policy, but data is fresh and kept near-policy.<br>**Mostly known for:** On-policy|
|PPO|✔|✔|With $K>1$ epochs it reuses data via a clipped importance ratio (technically off-policy), but keeps $\pi$ close to $\pi_{\text{old}}$.<br>**Mostly known for:** On-policy|
|DDPG|✔|✘|Deterministic policy gradient trained off a replay buffer.|
|TD3|✔|✘|DDPG with twin critics and target-policy smoothing; still replay-buffer based.|
|SAC|✔|✘|Maximum-entropy actor-critic trained from a replay buffer.|

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='full-episodes-intro'></a>

### Full Episodes vs Step-wise Updates (Algorithms' different behavior)

Algorithms differ in *how much* of a trajectory they observe before performing each update — anywhere from a single transition to a full episode.

$\cdot$ **Step-wise (bootstrapping)** — update after each step using the current value estimate. Lower variance, biased by the current estimate.

$\cdot$ **Fixed-length rollouts** — collect $n$ steps and then update; sits between the two extremes.

$\cdot$ **Complete episodes** — wait until the trajectory terminates before updating. Lower bias (true returns), higher variance.

| Algorithm | Updates from |
|---|---|
| Monte Carlo | Complete episodes |
| TD(0) | Each step (bootstraps) |
| $n$-step TD | $n$-step rollouts (full episodes only if $n = \infty$) |
| SARSA | Each step (bootstraps) |
| Q-learning | Each step (bootstraps) |
| REINFORCE | Complete episodes |
| PPO | Fixed-length rollouts (no need for full episodes) |

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='monte-carlo-first-visit'></a>

### Monte Carlo (First-Visit, On-Policy)

*Model-free, on-policy, learns from* ***complete*** *episodes.*

**Notation:**
- $G$ — cumulative discounted return (accumulated backwards from episode end)
- $T$ — terminal time step (total episode length)
- $\text{Returns}(s,a)$ — list of all observed returns for pair $(s,a)$ across all episodes so far

<a id='intro-2-1'></a>

**Introduction:** Model-free, on-policy control that learns $Q(s,a)$ from sample returns of ***complete*** episodes (no bootstrapping). *Idea:* run the current $\varepsilon$-greedy policy, record the actual discounted return $G_t$ from each (first-visit) state–action pair, and use the running average across episodes as the estimate $Q(s,a)$; then improve $\pi$ to be $\varepsilon$-greedy w.r.t. $Q$. *Why it helps:* unbiased — uses real returns rather than bootstrapped estimates — at the cost of high variance and needing episode termination.

**Algorithm:**

**Input:** number of episodes $n$, discount $\gamma$, exploration rate $\varepsilon$

1. Initialize $Q(s,a)$ arbitrarily for all $s \in S,  a \in A$; $\text{Returns}(s,a) \leftarrow \emptyset$; $\pi \leftarrow \varepsilon\text{-greedy w.r.t. } Q$

2. **For** episode $= 1, 2, \ldots, n$ *(outer loop — episodes)*:

   - **— Generate Episode —**

   - Generate a full episode using $\pi$: $\quad s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_{T-1}, a_{T-1}, r_T \quad \text{(until terminal)}$

   - **— Compute Returns and Update —**

   - $G \leftarrow 0$
   - **For** $t = T{-}1, T{-}2, \ldots, 0$ *(inner loop — steps, backwards)*:
     - $G \leftarrow \gamma G + r_{t+1}$
     - If $(s_t, a_t) \notin \lbrace (s_0,a_0), \ldots, (s_{t-1},a_{t-1})\rbrace$ *(first-visit check)*:
       - $\text{Returns}(s_t, a_t) \mathrel{+}= G$
       - $Q(s_t, a_t) \leftarrow \operatorname{mean}\bigl(\text{Returns}(s_t, a_t)\bigr)$
       - Update $\pi(s_t)$ to be $\varepsilon$-greedy w.r.t. $Q(s_t, \cdot)$:

$$\pi(a \mid s_t) = \begin{cases} 1 - \varepsilon + \dfrac{\varepsilon}{|A|} & \text{if } a = \arg\max_{a'} Q(s_t, a') \\[6pt] \dfrac{\varepsilon}{|A|} & \text{otherwise} \end{cases}$$

3. Return $Q,  \pi$

✦ ✦ ✦

**Notes:**

- Must wait until episode **ends** to learn (no bootstrapping).
- $G$ is the actual discounted return from step $t$ to the end of the episode.
- *First-visit*: only update $Q(s,a)$ using the return from the **first** time $(s,a)$ is visited in an episode. *Every-visit* MC updates for every occurrence.

<a id='tricks-2-1'></a>

**Additional Known Engineering Tricks**

- <a id='trick-2-1-1'></a>**Incremental mean update** — replace `Returns(s,a).append(G); Q ← mean(Returns)` with $Q(s,a)\leftarrow Q(s,a) + \frac{1}{N(s,a)}\bigl(G - Q(s,a)\bigr)$. *Why it helps:* removes the unbounded per-pair return list — constant memory per $(s,a)$ pair and constant-time updates instead of $O(N)$ averaging. *Interaction:* mathematically identical to the literal mean shown in the pseudocode; turning $\alpha = 1/N(s,a)$ into a *constant* $\alpha$ gives an exponentially-weighted moving average (non-stationary-friendly).

- **Exploring starts** *(Sutton & Barto, §5.3)* — start each episode at a uniformly random $(s_0, a_0)$ pair rather than the fixed initial state. *Why it helps:* guarantees every $(s,a)$ is sampled with positive probability under any policy, including deterministic greedy ones; otherwise on-policy MC can starve some pairs and never improve them. *Interaction:* an alternative to $\varepsilon$-greedy that allows learning a *deterministic* optimal policy directly; the two tricks are usually mutually exclusive.

- **GLIE schedule** *(Greedy in the Limit with Infinite Exploration)* — anneal $\varepsilon_k = 1/k$ (or $c/k$) with the episode count so that exploration vanishes in the limit but every action is still tried infinitely often. *Why it helps:* gives an *asymptotic* convergence guarantee to $\pi^*$ for on-policy MC control, which fixed $\varepsilon$ does not.

- <a id='trick-2-1-2'></a>**Every-visit MC** — update on *every* occurrence of $(s,a)$ in the episode, not only the first. *Why it helps:* lower variance estimator (more samples per episode) at the cost of a small bias that vanishes asymptotically. *Interaction:* the trick is binary with first-visit (you pick one); first-visit is preferred for theoretical clarity, every-visit for raw sample efficiency.

- <a id='trick-2-1-3'></a>**Weighted importance sampling (off-policy MC)** — collect episodes under a behaviour policy $\mu$, weight each return by $W=\prod_t \pi(a_t\mid s_t)/\mu(a_t\mid s_t)$, then use $Q\leftarrow \sum WG / \sum W$. *Why it helps:* lets you evaluate / improve a target $\pi$ from data generated by any (sufficiently exploratory) $\mu$; weighted form is biased but has bounded variance unlike ordinary IS.

<a id='pseudo-2-1'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 2.1</strong> First-visit Monte Carlo control (on-policy, $\varepsilon$-greedy) — Raw Pseudocode (without additional tricks) <em>[Sutton & Barto, §5.4]</em>
</div>

Here is the pseudocode for the first-visit Monte Carlo control core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** number of episodes $n$, discount $\gamma$, exploration rate $\varepsilon$  
Initialize $Q(s, a)$ arbitrarily for all $s \in S,  a \in A$  
$\text{Returns}(s, a) \leftarrow \emptyset$ for all $s, a$  
$\pi \leftarrow \varepsilon$-greedy w.r.t. $Q$  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Generate an episode using $\pi$: $s_0, a_0, r_1,  s_1, a_1, r_2,  \ldots,  s_{T-1}, a_{T-1}, r_T$  
&emsp;$G \leftarrow 0$  
&emsp;**for** $t = T - 1$ **down to** $0$ **do**  
&emsp;&emsp;$G \leftarrow \gamma  G + r_{t+1}$  
&emsp;&emsp;**if** $(s_t, a_t)$ does not appear in $(s_0, a_0), \ldots, (s_{t-1}, a_{t-1})$ **then**  
&emsp;&emsp;&emsp;Append $G$ to $\text{Returns}(s_t, a_t)$  
&emsp;&emsp;&emsp;$Q(s_t, a_t) \leftarrow \text{average}\bigl(\text{Returns}(s_t, a_t)\bigr)$  
&emsp;&emsp;&emsp;$\pi(\cdot \mid s_t) \leftarrow \varepsilon$-greedy w.r.t. $Q(s_t, \cdot)$  
&emsp;&emsp;**end if**  
&emsp;**end for**  
**end for**  
**Output:** $Q \approx q_{*}$, $\pi \approx \pi_{*}$  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 2.1</strong> First-visit Monte Carlo control (on-policy, $\varepsilon$-greedy) — Pseudocode with Additions (Engineering Tricks) <em>[Sutton & Barto, §5.4]</em>
</div>

Here is the pseudocode for the first-visit Monte Carlo control algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** number of episodes $n$, discount $\gamma$, ✦ initial $\varepsilon_1$ ✦ ([GLIE schedule](#trick-2-1-3))  
Initialize $Q(s, a)$ arbitrarily for all $s \in S,  a \in A$  
✦ Initialise $N(s,a) \leftarrow 0$ for all $s, a$ *(visit counts for incremental update)* ✦ ([Incremental mean update](#trick-2-1-1))  
$\pi \leftarrow \varepsilon$-greedy w.r.t. $Q$  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;✦ $\varepsilon \leftarrow 1/\text{episode}$ *(anneal toward greedy — GLIE)* ✦ ([GLIE schedule](#trick-2-1-3))  
&emsp;✦ Start episode at uniformly random $(s_0, a_0)$ *(alternative to $\varepsilon$-greedy)* ✦ ([Exploring starts](#trick-2-1-2))  
&emsp;Generate an episode using $\pi$: $s_0, a_0, r_1,  s_1, a_1, r_2,  \ldots,  s_{T-1}, a_{T-1}, r_T$  
&emsp;$G \leftarrow 0$  
&emsp;**for** $t = T - 1$ **down to** $0$ **do**  
&emsp;&emsp;$G \leftarrow \gamma  G + r_{t+1}$  
&emsp;&emsp;✦ *(Every-visit variant: skip the first-visit check and update on every occurrence)* ✦ ([Every-visit MC](#trick-2-1-4))  
&emsp;&emsp;**if** $(s_t, a_t)$ does not appear in $(s_0, a_0), \ldots, (s_{t-1}, a_{t-1})$ **then**  
&emsp;&emsp;&emsp;✦ $N(s_t,a_t) \leftarrow N(s_t,a_t) + 1$ ✦ ([Incremental mean update](#trick-2-1-1))  
&emsp;&emsp;&emsp;✦ $Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \frac{1}{N(s_t,a_t)}\bigl(G - Q(s_t,a_t)\bigr)$ *(replaces list-based mean)* ✦ ([Incremental mean update](#trick-2-1-1))  
&emsp;&emsp;&emsp;$\pi(\cdot \mid s_t) \leftarrow \varepsilon$-greedy w.r.t. $Q(s_t, \cdot)$  
&emsp;&emsp;**end if**  
&emsp;**end for**  
&emsp;✦ *(Off-policy variant: weight returns by $W=\prod_t \pi/\mu$, use $Q \leftarrow \sum WG / \sum W$)* ✦ ([Weighted importance sampling (off-policy MC)](#trick-2-1-5))  
**end for**  
**Output:** $Q \approx q_{*}$, $\pi \approx \pi_{*}$  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='td-0'></a>

### Temporal Difference — TD(0)

*Model-free; bootstraps after* ***each step*** *(no need to wait for episode end).*

**Notation:**
- $\alpha \in (0,1]$ — step size (learning rate)
- $s'$ — next state reached after taking action $a$ from state $s$

<a id='intro-2-2'></a>

**Introduction:** Model-free policy-evaluation method that estimates $V^\pi$ online by bootstrapping after ***each step*** (no need to wait for episode end). *Idea:* update $V(s) \leftarrow V(s) + \alpha\bigl[r + \gamma V(s') - V(s)\bigr]$, combining a single sampled reward with the current estimate of future value. *Why it helps over Monte Carlo:* learns online from incomplete episodes and has much lower variance, at the price of bias from the bootstrapped target.

**Algorithm:**

**Input:** policy $\pi$, step size $\alpha \in (0,1]$, discount $\gamma$, number of episodes $n$

1. Initialize $V(s)$ arbitrarily for all $s \in S$; $\quad V(\text{terminal}) = 0$

2. **For** episode $= 1, 2, \ldots, n$ *(outer loop — episodes)*:
   - Initialize $s$ (starting state)
   - **Repeat** for each step *(inner loop — time steps)*:
     - $a \leftarrow \pi(s)$
     - Take action $a$, observe $r,  s'$
     - $\displaystyle V(s) \leftarrow V(s) + \alpha\underbrace{\Bigl[r + \gamma V(s') - V(s)\Bigr]}_{\delta\ (\text{TD error})}$ *(TD update)*
     - $s \leftarrow s'$
   - Until $s'$ is terminal

3. Return $V$

✦ ✦ ✦

**Notes:**

- **TD target** $= r + \gamma V(s')$: a *biased* estimate of the true return, since $V(s')$ is itself an estimate — this is **bootstrapping**.
- **TD error** $\delta = r + \gamma V(s') - V(s)$: measures surprise — how much better the observed transition was vs. expectation.
- Unlike MC, TD can learn **before** the episode ends (online learning).
- Unlike DP, TD does **not** need the model $P(s' \mid s,a)$ — it samples transitions.
- This is *prediction only* (evaluating a given $\pi$). For control, see SARSA / Q-learning.

<a id='tricks-2-2'></a>

**Additional Known Engineering Tricks**

- **Eligibility traces — TD($\lambda$)** *(Sutton, 1988)* — maintain a per-state trace $e(s)$ that decays with $\gamma\lambda$, and update *all* states at every step: $V(s)\mathrel{+}=\alpha\delta e(s)$. *Why it helps:* propagates the TD error backwards through the recent trajectory in one shot, so credit reaches earlier states without waiting $n$ steps. *Interaction:* recovers TD(0) at $\lambda=0$ and Monte Carlo at $\lambda=1$ — bridges TD(0) and MC algorithmically, not just statistically.

- <a id='trick-2-2-1'></a>**Robbins–Monro step-size schedule** — choose $\alpha_t$ satisfying $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$, e.g. $\alpha_t = 1/(1+\text{visits}(s)/c)$ per-state. *Why it helps:* gives the formal almost-sure convergence guarantee for tabular TD(0); constant $\alpha$ converges only to a noise ball, not the fixed point.

- **Optimistic initial values** *(Sutton & Barto, §2.6)* — initialise $V(s)$ to a value strictly above the realistic return (e.g., $V(s)=R_\text{max}/(1-\gamma)$). *Why it helps:* every visited state will *decrease* its value relative to neighbours, so unvisited states look better and the policy is implicitly driven to explore — no $\varepsilon$ needed for evaluation under a greedy policy.

- **Gradient-TD (GTD/TDC)** *(Sutton et al., 2009)* — when combining TD(0) with linear function approximation under off-policy data, replace the semi-gradient with a true gradient of the projected Bellman error. *Why it helps:* fixes the *deadly triad* (off-policy + bootstrapping + function approximation) divergence problem; standard semi-gradient TD can diverge, GTD provably converges.

- <a id='trick-2-2-2'></a>**Linear semi-gradient TD(0)** — replace the tabular $V(s)$ with $\hat v(s;\mathbf{w}) = \mathbf{w}^\top \phi(s)$, update $\mathbf{w}\mathrel{+}=\alpha\delta\phi(s)$. *Why it helps:* scales TD to large/continuous state spaces by sharing parameters across states; this is the gateway from tabular RL to deep RL (where it becomes a neural net, see DQN).

- <a id='trick-2-2-3'></a>**Mini-batch / replay TD prediction** — store recent $(s,r,s')$ tuples in a buffer and sample mini-batches for offline TD updates. *Why it helps:* decorrelates samples for stochastic gradient methods and lets each transition be reused multiple times — same idea that powers DQN's experience replay, but applied to prediction.

<a id='pseudo-2-2'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 2.2</strong> Tabular TD(0) for estimating $v_\pi$ — Raw Pseudocode (without additional tricks) <em>[Sutton & Barto, §6.1]</em>
</div>

Here is the pseudocode for the tabular TD(0) core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** policy $\pi$, step size $\alpha \in (0, 1]$, discount $\gamma$, episodes $n$  
Initialize $V(s)$ arbitrarily for all $s \in S$; $V(\text{terminal}) \leftarrow 0$  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialize $s$  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;$a \leftarrow$ action given by $\pi$ for $s$  
&emsp;&emsp;Take action $a$; observe $r,  s'$  
&emsp;&emsp;$V(s) \leftarrow V(s) + \alpha \bigl[  r + \gamma  V(s') - V(s)  \bigr]$  
&emsp;&emsp;$s \leftarrow s'$  
&emsp;**until** $s$ is terminal  
**end for**  
**Output:** $V \approx v_{\pi}$  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 2.2</strong> Tabular TD(0) for estimating $v_\pi$ — Pseudocode with Additions (Engineering Tricks) <em>[Sutton & Barto, §6.1]</em>
</div>

Here is the pseudocode for the tabular TD(0) algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** policy $\pi$, ✦ initial step size $\alpha_0$ ✦ ([Robbins–Monro step-size schedule](#trick-2-2-2)), discount $\gamma$, ✦ trace decay $\lambda$ ✦ ([Eligibility traces — TD($\lambda$)](#trick-2-2-1)), episodes $n$  
✦ Initialise $V(s) \leftarrow R_\text{max}/(1-\gamma)$ for all $s$ *(optimistic)* ✦ ([Optimistic initial values](#trick-2-2-3))  
$V(\text{terminal}) \leftarrow 0$  
✦ *(Alternative: replace tabular $V(s)$ with $\hat v(s;\mathbf{w})=\mathbf{w}^\top\phi(s)$ — linear semi-gradient TD)* ✦ ([Linear semi-gradient TD(0)](#trick-2-2-5))  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialise $s$  
&emsp;✦ $e(s) \leftarrow 0$ for all $s$ *(eligibility traces)* ✦ ([Eligibility traces — TD($\lambda$)](#trick-2-2-1))  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;$a \leftarrow$ action given by $\pi$ for $s$  
&emsp;&emsp;Take action $a$; observe $r,  s'$  
&emsp;&emsp;$\delta \leftarrow r + \gamma  V(s') - V(s)$  
&emsp;&emsp;✦ $e(s) \leftarrow e(s) + 1$ *(accumulating trace)* ✦ ([Eligibility traces — TD($\lambda$)](#trick-2-2-1))  
&emsp;&emsp;✦ **for all** $\bar s \in S$ **do** $V(\bar s) \leftarrow V(\bar s) + \alpha \delta e(\bar s)$; $e(\bar s) \leftarrow \gamma\lambda e(\bar s)$ **end for** ✦ ([Eligibility traces — TD($\lambda$)](#trick-2-2-1))  
&emsp;&emsp;✦ $\alpha \leftarrow \alpha_0/(1 + \mathrm{visits}(s)/c)$ *(per-state Robbins–Monro decay)* ✦ ([Robbins–Monro step-size schedule](#trick-2-2-2))  
&emsp;&emsp;✦ *(Off-policy + function approx: use Gradient-TD / TDC update to guarantee convergence)* ✦ ([Gradient-TD (GTD/TDC)](#trick-2-2-4))  
&emsp;&emsp;$s \leftarrow s'$  
&emsp;**until** $s$ is terminal  
**end for**  
**Output:** $V \approx v_{\pi}$  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='n-step-td'></a>

### n-Step TD

*Model-free; bridges TD(0) and Monte Carlo — uses* ***$n$-step rollouts*** *(full episodes only if $n=\infty$).*

- $n=1$: equivalent to TD(0) (bootstrap after 1 step)
- $n=\infty$: equivalent to Monte Carlo (full episode return)

**Notation:**
- $n$ — lookahead depth (number of real steps before bootstrapping)
- $\tau$ — index of the state currently being updated; lags $n$ steps behind the current time $t$
- $T$ — terminal time step (recorded when episode ends)
- $G_{t:t+n}$ — $n$-step return starting at step $t$: $r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1}r_{t+n} + \gamma^n V(s_{t+n})$

<a id='intro-2-3'></a>

**Introduction:** Generalisation of TD(0) that bootstraps after $n$ real steps rather than 1. *Idea:* update toward the $n$-step return $G_{t:t+n} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1}r_{t+n} + \gamma^{n} V(s_{t+n})$, which interpolates between TD(0) ($n=1$, low variance / high bias) and Monte Carlo ($n=\infty$, no bootstrap, high variance / no bias). *Why it helps:* in practice a moderate $n$ trades bias against variance better than either extreme.

**Algorithm:**

**Input:** policy $\pi$, step size $\alpha$, discount $\gamma$, lookahead $n$, number of episodes $M$

1. Initialize $V(s)$ arbitrarily for all $s \in S$; $\quad V(\text{terminal}) = 0$

2. **For** episode $= 1, 2, \ldots, M$ *(outer loop — episodes)*:
   - Initialize $s_0$; $\quad T \leftarrow \infty$
   - **For** $t = 0, 1, 2, \ldots$ *(inner loop — time steps)*:
     - If $t < T$:
       - Take action $\pi(s_t)$, observe $r_{t+1},  s_{t+1}$
       - If $s_{t+1}$ is terminal: $T \leftarrow t+1$
     - Set $\tau \leftarrow t - n + 1$ *(time step being updated)*
     - If $\tau \geq 0$:
       - **Compute $n$-step return:**
         $$G \leftarrow \sum_{i=\tau+1}^{\min(\tau+n, T)} \gamma^{i-\tau-1}  r_i$$
       - If $\tau + n < T$: $G \leftarrow G + \gamma^n V(s_{\tau+n})$ *(bootstrap)*
       - **Update:** $\quad V(s_\tau) \leftarrow V(s_\tau) + \alpha \bigl[G - V(s_\tau)\bigr]$
   - Until $\tau = T - 1$

3. Return $V$

✦ ✦ ✦

**Notes:**

- The $n$-step return: $G_{t:t+n} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n})$
- Updates are *delayed* by $n$ steps (must wait $n$ steps before updating $s_t$).
- If the episode ends before $n$ steps, the return is a full MC return (no bootstrap).
- Larger $n$ $\Rightarrow$ lower bias, higher variance. $\quad$ Smaller $n$ $\Rightarrow$ higher bias, lower variance.

<a id='tricks-2-3'></a>

**Additional Known Engineering Tricks**

- **$\lambda$-return (forward view)** *(Sutton & Barto, §12.1)* — replace the single fixed-$n$ return by a geometric average $G_t^\lambda = (1-\lambda)\sum_{k\geq 1}\lambda^{k-1} G_{t:t+k}$. *Why it helps:* a soft mixture over all $n$-step returns gives a smoother bias–variance trade-off than picking one $n$; tuning $\lambda\in[0,1]$ is much less sensitive than tuning a discrete $n$. *Interaction:* the *backward-view* equivalent uses eligibility traces (next bullet), giving an $O(|S|)$-per-step online update.

- <a id='trick-2-3-1'></a>**Backward-view equivalence via eligibility traces** — instead of waiting $n$ steps and looking forward, maintain $e(s)$ traces and update *all* states at every step. *Why it helps:* removes the explicit $n$-step delay and the buffer of past states/rewards — online, $O(1)$-amortised update per step, identical fixed points to the forward view.

- **Tree-backup algorithm** *(Precup, 2000)* — for off-policy $n$-step without importance sampling, replace the bootstrap leaf with the expected value under the *target* policy and the action-probability of the on-policy action with the *behaviour* policy. *Why it helps:* enables off-policy $n$-step learning with bounded variance; classical $n$-step IS has variance that explodes with $n$.

- **Per-decision importance sampling** *(Precup et al., 2000)* — when off-policy, apply the IS ratio only on steps where the chosen action probability differs, instead of multiplying ratios over all $n$ steps. *Why it helps:* keeps the estimator unbiased while drastically reducing variance vs. ordinary IS.

- **Retrace($\lambda$)** *(Munos et al., 2016)* — clip the per-step IS ratio at 1 and combine with $\lambda$-returns: $c_k = \lambda\min(1, \pi/\mu)$. *Why it helps:* gives a *safe and efficient* off-policy $n$-step estimator — uniformly low variance, convergent for any $\mu$, and used in modern off-policy actor-critic (ACER, IMPALA).

- <a id='trick-2-3-2'></a>**Circular ring-buffer for the last $n$ transitions** — store $(s,r)$ in two fixed-size $n$-element arrays with a rolling index. *Why it helps:* removes per-step allocation churn; the inner $G\leftarrow \sum \gamma^{i}r$ becomes a single dot-product. *Interaction:* the pseudocode's references to $s_\tau$ and $r_i$ are abstract — in practice you cannot keep the full history, so a length-$n$ rolling buffer is the standard implementation.

<a id='pseudo-2-3'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 2.3</strong> $n$-step TD for estimating $v_\pi$ — Raw Pseudocode (without additional tricks) <em>[Sutton & Barto, §7.1]</em>
</div>

Here is the pseudocode for the $n$-step TD core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** policy $\pi$, step size $\alpha \in (0, 1]$, discount $\gamma$, lookahead $n$, episodes $M$  
Initialize $V(s)$ arbitrarily; $V(\text{terminal}) \leftarrow 0$  
**for** $\text{episode} = 1, 2, \ldots, M$ **do**  
&emsp;Initialize and store $s_0$; $T \leftarrow \infty$  
&emsp;**for** $t = 0, 1, 2, \ldots$ **do**  
&emsp;&emsp;**if** $t < T$ **then**  
&emsp;&emsp;&emsp;Take action $a_t \sim \pi(\cdot \mid s_t)$; observe $r_{t+1},  s_{t+1}$  
&emsp;&emsp;&emsp;**if** $s_{t+1}$ is terminal **then** $T \leftarrow t + 1$  
&emsp;&emsp;**end if**  
&emsp;&emsp;$\tau \leftarrow t - n + 1$ *(index of state being updated)*  
&emsp;&emsp;**if** $\tau \geq 0$ **then**  
&emsp;&emsp;&emsp;$G \leftarrow \displaystyle\sum_{i=\tau+1}^{\min(\tau+n,  T)} \gamma^{ i - \tau - 1}  r_i$  
&emsp;&emsp;&emsp;**if** $\tau + n < T$ **then** $G \leftarrow G + \gamma^{n}  V(s_{\tau + n})$  
&emsp;&emsp;&emsp;$V(s_\tau) \leftarrow V(s_\tau) + \alpha \bigl[  G - V(s_\tau)  \bigr]$  
&emsp;&emsp;**end if**  
&emsp;**until** $\tau = T - 1$  
**end for**  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 2.3</strong> $n$-step TD for estimating $v_\pi$ — Pseudocode with Additions (Engineering Tricks) <em>[Sutton & Barto, §7.1]</em>
</div>

Here is the pseudocode for the $n$-step TD algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** policy $\pi$, step size $\alpha$, discount $\gamma$, ✦ trace decay $\lambda$ ✦ ([λ-return (forward view)](#trick-2-3-1)), lookahead $n$, episodes $M$  
Initialise $V(s)$ arbitrarily; $V(\text{terminal}) \leftarrow 0$  
**for** $\text{episode} = 1, 2, \ldots, M$ **do**  
&emsp;Initialise and store $s_0$; $T \leftarrow \infty$  
&emsp;✦ Allocate circular ring-buffer $\mathrm{buf}[0\ldots n{-}1]$ for $(s, r)$ pairs ✦ ([Circular ring-buffer for the last $n$ transitions](#trick-2-3-6))  
&emsp;✦ $e(s) \leftarrow 0$ for all $s$ *(eligibility traces — backward-view equivalent)* ✦ ([Backward-view equivalence via eligibility traces](#trick-2-3-2))  
&emsp;**for** $t = 0, 1, 2, \ldots$ **do**  
&emsp;&emsp;**if** $t < T$ **then**  
&emsp;&emsp;&emsp;Take action $a_t \sim \pi(\cdot \mid s_t)$; observe $r_{t+1},  s_{t+1}$  
&emsp;&emsp;&emsp;✦ Store $(s_t, r_{t+1})$ in $\mathrm{buf}[t \bmod n]$ ✦ ([Circular ring-buffer for the last $n$ transitions](#trick-2-3-6))  
&emsp;&emsp;&emsp;**if** $s_{t+1}$ is terminal **then** $T \leftarrow t + 1$  
&emsp;&emsp;**end if**  
&emsp;&emsp;$\tau \leftarrow t - n + 1$  
&emsp;&emsp;**if** $\tau \geq 0$ **then**  
&emsp;&emsp;&emsp;✦ $G \leftarrow G^\lambda_\tau = (1-\lambda)\sum_{k\geq 1}\lambda^{k-1} G_{\tau:\tau+k}$ *($\lambda$-return mixing all $n$-step targets)* ✦ ([λ-return (forward view)](#trick-2-3-1))  
&emsp;&emsp;&emsp;✦ *(Off-policy: apply per-decision IS ratios $\rho_i = \pi(a_i\mid s_i)/\mu(a_i\mid s_i)$ to each step)* ✦ ([Per-decision importance sampling](#trick-2-3-4))  
&emsp;&emsp;&emsp;✦ *(Off-policy alternative: use Retrace($\lambda$) with $c_k = \lambda\min(1,\pi/\mu)$ for safe correction)* ✦ ([Retrace($\lambda$)](#trick-2-3-5))  
&emsp;&emsp;&emsp;✦ *(Off-policy alternative: tree-backup — no IS ratios, replace leaf with $\mathbb{E}_\pi[Q]$)* ✦ ([Tree-backup algorithm](#trick-2-3-3))  
&emsp;&emsp;&emsp;$V(s_\tau) \leftarrow V(s_\tau) + \alpha \bigl[  G - V(s_\tau)  \bigr]$  
&emsp;&emsp;&emsp;✦ *(Backward-view equivalent: update all states via traces $e(\bar s) \leftarrow \gamma\lambda e(\bar s)$; $V(\bar s)\mathrel{+}=\alpha\delta e(\bar s)$)* ✦ ([Backward-view equivalence via eligibility traces](#trick-2-3-2))  
&emsp;&emsp;**end if**  
&emsp;**until** $\tau = T - 1$  
**end for**  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='sarsa'></a>

### SARSA (State-Action-Reward-State-Action)

*Model-free, on-policy, TD(0)-based control (learns $Q$ directly); bootstraps after* ***each step*** *(no need for full episodes).*
*Named after the quintuple: $(s, a, r, s', a')$.*

**Notation:**
- $\alpha$ — step size (learning rate)
- $s'$ — next state after taking $a$
- $a'$ — next action, chosen by the current policy from $s'$ *before* the update is applied (hence the name SARSA)

<a id='intro-2-4'></a>

**Introduction:** Model-free, ***on-policy*** TD control. Named after the quintuple $(s, a, r, s', a')$ that drives each update. *Idea:* update $Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[r + \gamma  Q(s', a') - Q(s,a)\bigr]$ using the action $a'$ that the **current** ($\varepsilon$-greedy) policy actually chooses from $s'$. *Why it helps:* on-policy estimates account for exploration in the target, so SARSA learns the value of the policy it actually follows — preferring safer paths than Q-learning when exploration noise is non-trivial (the canonical cliff-walking example).

**Algorithm:**

**Input:** step size $\alpha \in (0,1]$, discount $\gamma$, exploration rate $\varepsilon$, episodes $n$

1. Initialize $Q(s,a)$ arbitrarily for all $s \in S,  a \in A$; $\quad Q(\text{terminal}, \cdot) = 0$

2. **For** episode $= 1, 2, \ldots, n$ *(outer loop — episodes)*:
   - Initialize $s$; choose $a$ from $s$ using $\varepsilon$-greedy w.r.t. $Q$
   - **Repeat** for each step *(inner loop — time steps)*:
     - Take action $a$, observe $r,  s'$
     - Choose $a'$ from $s'$ using $\varepsilon$-greedy w.r.t. $Q$ *(choose next action now)*
     - $Q(s,a) \leftarrow Q(s,a) + \alpha \bigl[r + \gamma  Q(s',a') - Q(s,a)\bigr]$  *(uses $a'$ from the **same** policy — on-policy)*
     - $s \leftarrow s'$; $\quad a \leftarrow a'$
   - Until $s$ is terminal

3. Return $Q$; derive $\pi(s) = \arg\max_a Q(s,a)$

✦ ✦ ✦

**Notes:**

- **On-policy:** the action $a'$ used in the TD target $Q(s',a')$ is chosen by the **same** $\varepsilon$-greedy policy that the agent is following. The agent learns about the policy it is actually using (including its exploration).
- Because it accounts for exploration, SARSA tends to learn **safer / more conservative** policies than Q-learning in stochastic environments.
- The update uses $Q(s',a')$ where $a'$ is already chosen — hence the name $s, a, r, s', a'$.

<a id='tricks-2-4'></a>

**Additional Known Engineering Tricks**

- **Expected SARSA** *(van Seijen et al., 2009)* — replace the sampled $Q(s',a')$ in the TD target with the expectation under $\pi$: $y = r + \gamma\sum_{a'}\pi(a'\mid s') Q(s',a')$. *Why it helps:* removes the sampling variance of $a'$ at no bias cost — a strict improvement over standard SARSA for the same number of samples. *Interaction:* recovers Q-learning when $\pi$ is greedy and recovers SARSA when $\pi$ is the sampling policy.

- <a id='trick-2-4-1'></a>**SARSA($\lambda$)** — eligibility traces over $(s,a)$ pairs, decayed by $\gamma\lambda$; on each step update *every* $(s,a)$ visited recently by $\alpha\delta e(s,a)$. *Why it helps:* much faster credit assignment over long episodes than one-step SARSA. *Interaction:* watch out — combining with $\varepsilon$-greedy makes the trace technically off-policy on exploratory actions; the common Watkins-style fix is to *cut* the trace whenever a non-greedy action is taken.

- <a id='trick-2-4-2'></a>**Boltzmann (softmax) exploration** — sample $a\sim\mathrm{softmax}(Q(s,\cdot)/\tau)$ instead of $\varepsilon$-greedy. *Why it helps:* allocates exploration in proportion to perceived value rather than uniformly over non-greedy actions; usually safer for SARSA's *on-policy* value, which directly reflects the cost of suboptimal exploration. *Interaction:* mutually exclusive with $\varepsilon$-greedy; temperature $\tau$ is annealed similarly to $\varepsilon$.

- <a id='trick-2-4-3'></a>**n-step SARSA** — use the $n$-step on-policy return $G_{t:t+n} = r_{t+1}+\dots+\gamma^{n-1}r_{t+n}+\gamma^n Q(s_{t+n}, a_{t+n})$ as the target. *Why it helps:* the same bias-variance trade-off as $n$-step TD prediction, but for control; smooths SARSA's reaction time without forcing full Monte-Carlo episodes.

- <a id='trick-2-4-4'></a>**Double SARSA / Double Expected SARSA** — maintain $Q^A$, $Q^B$; on each step randomly pick one to update using the other's value at $(s',a')$. *Why it helps:* decouples action selection from action evaluation, breaking the maximisation-bias problem when $Q$ contains noise; same idea as Double Q-learning but for the on-policy target.

- <a id='trick-2-4-5'></a>**Optimistic initialisation** — set $Q(s,a)$ high so unvisited pairs look attractive. *Why it helps:* drives systematic exploration without explicit $\varepsilon$; especially effective in SARSA because the on-policy target dampens the optimism gracefully as actions are tried.

<a id='pseudo-2-4'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 2.4</strong> SARSA (on-policy TD control) — Raw Pseudocode (without additional tricks) <em>[Sutton & Barto, §6.4]</em>
</div>

Here is the pseudocode for the SARSA core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** step size $\alpha \in (0, 1]$, discount $\gamma$, exploration rate $\varepsilon$, episodes $n$  
Initialize $Q(s, a)$ arbitrarily for all $s \in S,  a \in A$; $Q(\text{terminal}, \cdot) = 0$  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialize $s$  
&emsp;Choose $a$ from $s$ using $\varepsilon$-greedy w.r.t. $Q$  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;Take action $a$; observe $r,  s'$  
&emsp;&emsp;Choose $a'$ from $s'$ using $\varepsilon$-greedy w.r.t. $Q$  
&emsp;&emsp;$Q(s, a) \leftarrow Q(s, a) + \alpha \bigl[  r + \gamma  Q(s', a') - Q(s, a)  \bigr]$  
&emsp;&emsp;$s \leftarrow s';\quad a \leftarrow a'$  
&emsp;**until** $s$ is terminal  
**end for**  
**Output:** $Q \approx q_{*}$  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 2.4</strong> SARSA (on-policy TD control) — Pseudocode with Additions (Engineering Tricks) <em>[Sutton & Barto, §6.4]</em>
</div>

Here is the pseudocode for the SARSA algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** step size $\alpha$, discount $\gamma$, ✦ temperature $\tau$ ✦ ([Boltzmann (softmax) exploration](#trick-2-4-3)), ✦ trace decay $\lambda$ ✦ ([SARSA($\lambda$)](#trick-2-4-2)), ✦ lookahead $n$ ✦ ([n-step SARSA](#trick-2-4-4)), episodes $M$  
✦ Initialise $Q^A(s,a), Q^B(s,a) \leftarrow R_\text{max}/(1-\gamma)$ for all $s, a$ *(optimistic; two tables for Double SARSA)* ✦ ([Optimistic initialisation](#trick-2-4-6) / [Double SARSA / Double Expected SARSA](#trick-2-4-5))  
$Q(\text{terminal}, \cdot) = 0$  
**for** $\text{episode} = 1, 2, \ldots, M$ **do**  
&emsp;Initialise $s$  
&emsp;✦ $e(s,a) \leftarrow 0$ for all $s, a$ *(eligibility traces)* ✦ ([SARSA($\lambda$)](#trick-2-4-2))  
&emsp;✦ Choose $a \sim \mathrm{softmax}(Q(s,\cdot)/\tau)$ ✦ ([Boltzmann (softmax) exploration](#trick-2-4-3))  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;Take action $a$; observe $r,  s'$  
&emsp;&emsp;✦ Choose $a' \sim \mathrm{softmax}(Q(s',\cdot)/\tau)$ ✦ ([Boltzmann (softmax) exploration](#trick-2-4-3))  
&emsp;&emsp;✦ *(n-step variant: accumulate $G_{t:t+n} = \sum_{k=0}^{n-1}\gamma^k r_{t+k+1} + \gamma^n Q(s_{t+n},a_{t+n})$ before updating)* ✦ ([n-step SARSA](#trick-2-4-4))  
&emsp;&emsp;✦ $y \leftarrow r + \gamma\sum_{a''}\pi(a''\mid s')Q(s',a'')$ *(Expected SARSA target — replaces sampled $Q(s',a')$)* ✦ ([Expected SARSA](#trick-2-4-1))  
&emsp;&emsp;$\delta \leftarrow y - Q(s, a)$  
&emsp;&emsp;✦ $e(s,a) \leftarrow e(s,a) + 1$ ✦ ([SARSA($\lambda$)](#trick-2-4-2))  
&emsp;&emsp;✦ **for all** $(\bar s,\bar a)$ **do** $Q(\bar s,\bar a) \leftarrow Q(\bar s,\bar a) + \alpha \delta e(\bar s,\bar a)$; $e(\bar s,\bar a) \leftarrow \gamma\lambda e(\bar s,\bar a)$ **end for** ✦ ([SARSA($\lambda$)](#trick-2-4-2))  
&emsp;&emsp;✦ *(Double variant: with 50 % prob. update $Q^A$ using $Q^B$'s value at $(s',a')$, else vice versa)* ✦ ([Double SARSA / Double Expected SARSA](#trick-2-4-5))  
&emsp;&emsp;$s \leftarrow s';\quad a \leftarrow a'$  
&emsp;**until** $s$ is terminal  
**end for**  
**Output:** $Q \approx q_{*}$  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='q-learning'></a>

### Q-Learning

*Model-free, off-policy, TD(0)-based control (learns $Q$ directly); bootstraps after* ***each step*** *(no need for full episodes).*

**Notation:**
- $\alpha$ — step size (learning rate)
- $s'$ — next state after taking $a$
- $a'$ — dummy variable ranging over all actions at $s'$; only used inside $\max_{a'}$ and never actually taken

<a id='intro-2-5'></a>

**Introduction:** Model-free, ***off-policy*** TD control — learns the optimal action-value function $Q^*$ regardless of the behaviour policy. *Idea:* update $Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[r + \gamma \max_{a'} Q(s', a') - Q(s,a)\bigr]$ using the ***greedy*** action at $s'$, even though the behaviour policy may have explored elsewhere. *Why it helps:* decouples behaviour and target — the agent can explore widely (or follow any policy that covers $S\times A$) while the learnt $Q$ still converges to $Q^*$.

**Algorithm:**

**Input:** step size $\alpha \in (0,1]$, discount $\gamma$, exploration rate $\varepsilon$, episodes $n$

1. Initialize $Q(s,a)$ arbitrarily for all $s \in S,  a \in A$; $\quad Q(\text{terminal}, \cdot) = 0$

2. **For** episode $= 1, 2, \ldots, n$ *(outer loop — episodes)*:
   - Initialize $s$
   - **Repeat** for each step *(inner loop — time steps)*:
     - Choose $a$ from $s$ using $\varepsilon$-greedy w.r.t. $Q$ *(behavior policy — explores)*
     - Take action $a$, observe $r,  s'$
     - $Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]$  *(uses **greedy max**, not the action actually taken)*
     - $s \leftarrow s'$
   - Until $s$ is terminal

3. Return $Q$; derive $\pi(s) = \arg\max_a Q(s,a)$

✦ ✦ ✦

**Notes:**

- **Off-policy:** the agent follows an $\varepsilon$-greedy behavior policy (to explore), but the TD target uses $\max_{a'} Q(s',a')$ — the **greedy** action — regardless of what the agent actually takes next. Q-learning directly learns $Q^*$ even while exploring.
- **Key difference from SARSA:**

| | TD update target |
|---|---|
| SARSA | $r + \gamma  Q(s', a')$ where $a' \sim \varepsilon\text{-greedy}$ |
| Q-learning | $r + \gamma \max_{a'} Q(s', a')$ (greedy max) |

- Q-learning can be more sample-efficient but may learn riskier policies than SARSA in stochastic environments (it ignores the cost of exploration).

<a id='tricks-2-5'></a>

**Additional Known Engineering Tricks**

- **Double Q-learning** *(van Hasselt, 2010)* — maintain two value tables $Q^A, Q^B$; on each step, with 50% probability update $Q^A(s,a)\leftarrow Q^A(s,a) + \alpha[r + \gamma Q^B(s', \arg\max_{a'} Q^A(s',a')) - Q^A(s,a)]$ (and symmetrically). *Why it helps:* the $\max_{a'}$ in standard Q-learning systematically overestimates the action value because of noise; using a *different* table to evaluate the action selected by the first cancels this bias. *Interaction:* foundational idea behind Double DQN in deep RL.

- **Watkins's Q($\lambda$) / Peng's Q($\lambda$)** *(Watkins, 1989; Peng & Williams, 1996)* — eligibility traces for off-policy Q-learning. Watkins's variant *cuts* the trace to zero whenever a non-greedy action is taken (preserves convergence); Peng's leaves the trace alive (faster but biased). *Why it helps:* propagates credit over multi-step trajectories without waiting; Watkins is the safe default. *Interaction:* the trace-cut is necessary precisely *because* Q-learning is off-policy — see also the *retrace* fix in the $n$-step TD section.

- **Tile coding / state aggregation** *(CMAC, Albus 1975)* — partition continuous $s$ into overlapping tilings of bins and use the indicator vector as features; $Q$ becomes a weighted sum over active tiles. *Why it helps:* a simple, theoretically-grounded linear function approximation that scales tabular Q-learning to medium-dimensional continuous state spaces with provable convergence (for on-policy update; care needed off-policy).

- <a id='trick-2-5-1'></a>**Count-based / UCB-style exploration** — replace $\varepsilon$-greedy with $a\leftarrow\arg\max_a [Q(s,a) + c\sqrt{\log t / N(s,a)}]$, where $N(s,a)$ is the visit count. *Why it helps:* directs exploration to truly under-sampled actions instead of random ones, dramatically improving sample efficiency in sparse-reward MDPs.

- **Reward shaping** *(Ng, Harada & Russell, 1999)* — add a *potential-based* shaping reward $F(s,a,s') = \gamma\Phi(s')-\Phi(s)$ to the environment reward. *Why it helps:* injects domain knowledge to densify the reward signal without changing the optimal policy (the potential telescopes out). Critically, only *potential-based* shaping is policy-invariant — arbitrary shaping can change the optimum.

- <a id='trick-2-5-2'></a>**Optimistic $Q$ initialisation** — initialise $Q(s,a) = R_\text{max}/(1-\gamma)$. *Why it helps:* drives early exploration to under-visited actions even with greedy selection; less wasteful than $\varepsilon$-greedy on tasks with a clear gradient of reward.

<a id='pseudo-2-5'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 2.5</strong> Q-learning (off-policy TD control) — Raw Pseudocode (without additional tricks) <em>[Sutton & Barto, §6.5; Watkins, 1989]</em>
</div>

Here is the pseudocode for the Q-learning core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** step size $\alpha \in (0, 1]$, discount $\gamma$, exploration rate $\varepsilon$, episodes $n$  
Initialize $Q(s, a)$ arbitrarily for all $s \in S,  a \in A$; $Q(\text{terminal}, \cdot) = 0$  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialize $s$  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;Choose $a$ from $s$ using $\varepsilon$-greedy w.r.t. $Q$ *(behavior policy)*  
&emsp;&emsp;Take action $a$; observe $r,  s'$  
&emsp;&emsp;$Q(s, a) \leftarrow Q(s, a) + \alpha \bigl[  r + \gamma  \max_{a'} Q(s', a') - Q(s, a)  \bigr]$  
&emsp;&emsp;$s \leftarrow s'$  
&emsp;**until** $s$ is terminal  
**end for**  
**Output:** $Q \approx q_{*}$  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 2.5</strong> Q-learning (off-policy TD control) — Pseudocode with Additions (Engineering Tricks) <em>[Sutton & Barto, §6.5; Watkins, 1989]</em>
</div>

Here is the pseudocode for the Q-learning algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** step size $\alpha$, discount $\gamma$, ✦ trace decay $\lambda$ ✦ ([Watkins's Q($\lambda$) / Peng's Q($\lambda$)](#trick-2-5-2)), ✦ UCB constant $c$ ✦ ([Count-based / UCB-style exploration](#trick-2-5-4)), ✦ potential function $\Phi(s)$ ✦ ([Reward shaping](#trick-2-5-5)), episodes $n$  
✦ Initialise $Q^A(s,a), Q^B(s,a) \leftarrow R_\text{max}/(1-\gamma)$ *(optimistic; two tables for Double Q)* ✦ ([Optimistic $Q$ initialisation](#trick-2-5-6) / [Double Q-learning](#trick-2-5-1))  
$Q(\text{terminal}, \cdot) = 0$  
✦ $N(s,a) \leftarrow 0$ for all $s, a$ *(visit counts for UCB)* ✦ ([Count-based / UCB-style exploration](#trick-2-5-4))  
✦ *(Alternative: replace tabular $Q$ with tile coding — $Q(s,a)=\mathbf{w}^\top\phi(s,a)$ over overlapping tilings)* ✦ ([Tile coding / state aggregation](#trick-2-5-3))  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialise $s$  
&emsp;✦ $e(s,a) \leftarrow 0$ for all $s, a$ *(eligibility traces)* ✦ ([Watkins's Q($\lambda$) / Peng's Q($\lambda$)](#trick-2-5-2))  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;✦ $a \leftarrow \arg\max_a \bigl[Q(s,a) + c\sqrt{\log t / N(s,a)}\bigr]$ *(UCB action selection)* ✦ ([Count-based / UCB-style exploration](#trick-2-5-4))  
&emsp;&emsp;Take action $a$; observe $r,  s'$  
&emsp;&emsp;✦ $N(s,a) \leftarrow N(s,a) + 1$ ✦ ([Count-based / UCB-style exploration](#trick-2-5-4))  
&emsp;&emsp;✦ $r' \leftarrow r + \gamma \Phi(s') - \Phi(s)$ *(potential-based reward shaping)* ✦ ([Reward shaping](#trick-2-5-5))  
&emsp;&emsp;✦ With prob. 0.5: $a^*\leftarrow\arg\max_{a'} Q^A(s',a')$; $Q^A(s,a)\leftarrow Q^A(s,a) + \alpha\bigl[r' + \gamma  Q^B(s',a^*) - Q^A(s,a)\bigr]$; else swap A/B ✦ ([Double Q-learning](#trick-2-5-1))  
&emsp;&emsp;✦ $e(s,a) \leftarrow e(s,a)+1$; **if** $a$ was greedy: decay $e \leftarrow \gamma\lambda e$; **else** cut $e \leftarrow 0$ *(Watkins)* ✦ ([Watkins's Q($\lambda$) / Peng's Q($\lambda$)](#trick-2-5-2))  
&emsp;&emsp;✦ **for all** $(\bar s,\bar a)$ **do** $Q(\bar s,\bar a)\mathrel{+}=\alpha \delta e(\bar s,\bar a)$ **end for** ✦ ([Watkins's Q($\lambda$) / Peng's Q($\lambda$)](#trick-2-5-2))  
&emsp;&emsp;$s \leftarrow s'$  
&emsp;**until** $s$ is terminal  
**end for**  
**Output:** $Q \approx q_{*}$  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

### Summary facts

![image.png](attachment:image.png) ![image-2.png](attachment:image-2.png)

✦ ✦ ✦
## Deep RL — Value-Based
✦ ✦ ✦

<a id='dqn-naive'></a>

### DQN Naive (Deep Q-Network, no tricks)

**Notation:**
- $\theta$ — neural network weights (learnable parameters; *not* a convergence threshold here)
- $Q(s,a;\theta)$ — Q-value approximated by a neural net with weights $\theta$
- $\mathcal{L}(\theta)$ — loss function (squared Bellman error)
- $\nabla_\theta$ — gradient with respect to $\theta$

<a id='intro-3-1'></a>

**Introduction:** Q-learning with a neural network $Q(s,a;\theta)$ in place of the tabular Q. *Idea:* minimise the squared one-step Bellman error $\mathcal{L}(\theta) = \bigl(r + \gamma \max_{a'} Q(s', a';\theta) - Q(s, a;\theta)\bigr)^2$ via stochastic gradient descent. *Why it helps:* scales Q-learning to large/continuous state spaces (e.g. pixel inputs) where a Q-table is infeasible — but this naive form is unstable because the target and the prediction share the same rapidly changing weights $\theta$.

**Algorithm:**

**Input:** step size $\alpha$, discount $\gamma$, exploration rate $\varepsilon$, episodes $n$

1. Initialize neural network $Q(s,a;\theta)$ with random weights $\theta$

2. **For** episode $= 1, 2, \ldots, n$ *(outer loop — episodes)*:
   - Initialize $s$
   - **Repeat** for each step *(inner loop — time steps)*:
     - Choose $a$ using $\varepsilon$-greedy w.r.t. $Q(s,\cdot;\theta)$
     - Take action $a$, observe $r,  s'$
     - **Compute target** *(uses the **same** network $\theta$)*:
       $$y \leftarrow r + \gamma \max_{a'} Q(s', a';  \theta)$$
     - **Gradient update:**
       $$\mathcal{L}(\theta) = \bigl(y - Q(s,a;\theta)\bigr)^2, \qquad \theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$$
     - $s \leftarrow s'$
   - Until $s$ is terminal

3. Return $Q(\cdot,\cdot;\theta)$; derive $\pi(s) = \arg\max_a Q(s,a;\theta)$

✦ ✦ ✦

**Problems with Naive DQN:**

- **Instability:** the target $y$ depends on the same $\theta$ being updated — every gradient step shifts both the prediction and the target $\Rightarrow$ *moving target problem*.
- **Correlated samples:** consecutive transitions $(s,a,r,s')$ are highly correlated; SGD assumes i.i.d. samples $\Rightarrow$ this assumption is violated.
- These two issues cause divergence or oscillation in practice.

$\Rightarrow$ Fixed by **Target Network** and **Experience Replay** (see below).

<a id='tricks-3-1'></a>

**Additional Known Engineering Tricks**

- **Reward clipping to $[-1, +1]$** *(Mnih et al., 2013)* — clip the per-step reward into a fixed range before the TD target is computed. *Why it helps:* puts all Atari games (or other tasks with wildly different reward magnitudes) on a single learning-rate scale; prevents a single jackpot reward from blowing up the gradient. *Interaction:* changes the *value of $\gamma$ effectively* (since rewards are bounded), so combine with the standard $\gamma=0.99$.

- <a id='trick-3-1-1'></a>**Frame skipping / action repeat** — repeat the same action for $k=4$ environment steps and only call the network on every $k$-th frame. *Why it helps:* cuts the inference cost by $k$ and increases the effective discount horizon at no quality cost in environments with high temporal correlation between successive frames.

- <a id='trick-3-1-2'></a>**Frame stacking** — concatenate the last $k=4$ pre-processed frames into a single $k$-channel observation. *Why it helps:* injects a velocity signal into an otherwise-Markovian-in-pixels state; the Q-network can then infer motion without recurrence. *Interaction:* combines tightly with frame skipping — the stack contains $k$ skipped frames, giving a $k^2$-frame temporal window.

- <a id='trick-3-1-3'></a>**Atari pre-processing pipeline** — grayscale, resize to $84\times 84$, and take the *pixel-wise max* over two adjacent raw frames to handle Atari's odd–even sprite flickering. *Why it helps:* drops input dimensionality by $\sim 30\times$ and removes sprite-flicker noise that would otherwise look like a non-Markovian observation.

- **Huber loss (smooth L1)** instead of MSE — quadratic for $|y - Q|\le 1$, linear beyond. *Why it helps:* equivalent to clipping the TD error gradient at $\pm 1$ — bounds the magnitude of a single bad target so a noisy $y$ cannot blow up the parameters of the value head.

- **RMSProp with low learning rate** *(Mnih et al., 2013 — $\alpha=2.5\times10^{-4}$, momentum $0.95$, $\epsilon=10^{-2}$)* — explicit per-parameter step normalisation. *Why it helps:* compensates for highly-variable gradient scales across the conv stack; the specific small-$\alpha$ + large-$\epsilon$ combo is what made the *naive* (no target net, no replay) DQN converge at all on simple Atari games. *Interaction:* once target net + replay are added (next sections), Adam with $\alpha\sim10^{-4}$ works fine and supersedes this choice.

<a id='pseudo-3-1'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 3.1</strong> DQN (naive — no target net, no replay) — Raw Pseudocode (without additional tricks) <em>[Mnih et al., 2013]</em>
</div>

Here is the pseudocode for the DQN (naive) core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** step size $\alpha$, discount $\gamma$, exploration rate $\varepsilon$, episodes $n$  
Initialize $Q$-network $Q(s, a;  \theta)$ with random weights $\theta$  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialize $s$  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;Choose $a$ from $s$ using $\varepsilon$-greedy w.r.t. $Q(s, \cdot;  \theta)$  
&emsp;&emsp;Take action $a$; observe $r,  s'$  
&emsp;&emsp;$y \leftarrow r + \gamma  \max_{a'} Q(s', a';  \theta)$ *(bootstrap from same network)*  
&emsp;&emsp;$\mathcal{L}(\theta) \leftarrow \bigl(  y - Q(s, a;  \theta)  \bigr)^{2}$  
&emsp;&emsp;$\theta \leftarrow \theta - \alpha  \nabla_{\theta}  \mathcal{L}(\theta)$  
&emsp;&emsp;$s \leftarrow s'$  
&emsp;**until** $s$ is terminal  
**end for**  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 3.1</strong> DQN (naive — no target net, no replay) — Pseudocode with Additions (Engineering Tricks) <em>[Mnih et al., 2013]</em>
</div>

Here is the pseudocode for the DQN (naive) algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** ✦ RMSProp with $\alpha=2.5\times10^{-4}$, momentum $0.95$, $\epsilon_\text{opt}=10^{-2}$ ✦ ([RMSProp with low learning rate](#trick-3-1-6)), discount $\gamma$, exploration $\varepsilon$, episodes $n$, ✦ action repeat $k=4$ ✦ ([Frame skipping / action repeat](#trick-3-1-2))  
Initialise $Q$-network $Q(s, a;  \theta)$ with random weights $\theta$  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialise $s$  
&emsp;✦ Pre-process $s$: grayscale, resize to $84\times84$, pixel-max over last 2 raw frames ✦ ([Atari pre-processing pipeline](#trick-3-1-4))  
&emsp;✦ Stack last $k=4$ pre-processed frames as channels ✦ ([Frame stacking](#trick-3-1-3))  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;Choose $a$ from $s$ using $\varepsilon$-greedy w.r.t. $Q(s, \cdot;  \theta)$  
&emsp;&emsp;✦ Repeat action $a$ for $k=4$ environment steps; accumulate reward ✦ ([Frame skipping / action repeat](#trick-3-1-2))  
&emsp;&emsp;Take action $a$; observe $r,  s'$  
&emsp;&emsp;✦ $r \leftarrow \mathrm{clip}(r,  -1,  +1)$ ✦ ([Reward clipping to $[-1,+1]$](#trick-3-1-1))  
&emsp;&emsp;✦ Pre-process and frame-stack $s'$ ✦ ([Atari pre-processing pipeline](#trick-3-1-4) / [Frame stacking](#trick-3-1-3))  
&emsp;&emsp;$y \leftarrow r + \gamma  \max_{a'} Q(s', a';  \theta)$  
&emsp;&emsp;✦ $\mathcal{L}(\theta) \leftarrow \mathrm{Huber}\bigl(y - Q(s, a;  \theta)\bigr)$ *(smooth L1 — linear for $|e|>1$, quadratic otherwise)* ✦ ([Huber loss (smooth L1)](#trick-3-1-5))  
&emsp;&emsp;✦ $\theta \leftarrow \theta - \mathrm{RMSProp}(\nabla_\theta  \mathcal{L})$ ✦ ([RMSProp with low learning rate](#trick-3-1-6))  
&emsp;&emsp;$s \leftarrow s'$  
&emsp;**until** $s$ is terminal  
**end for**  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='dqn-target-network'></a>

### DQN + Target Network (DQN+TN)

**Notation:**
- $\theta^-$ — frozen target network weights (a periodic hard copy of $\theta$; never updated by gradients)
- $\hat{Q}(s,a;\theta^-)$ — target network (separate from the online network $Q$)
- $C$ — target update frequency (gradient steps between hard copies $\theta^- \leftarrow \theta$)
- $t$ — global step counter

<a id='intro-3-2'></a>

**Introduction:** DQN with a ***separate frozen target network*** $\hat{Q}(s,a;\theta^-)$ used to compute the Bellman target; $\theta^-$ is a periodic hard copy of the online weights $\theta$. *Idea:* compute the target with $\theta^-$ instead of $\theta$ so the target stays still while the online network chases it, then sync $\theta^- \leftarrow \theta$ every $C$ steps. *Why it helps:* fixes the moving-target problem that destabilises naive DQN — divergence/oscillation are replaced by a steady (if slightly stale) regression target.

**Algorithm:**

**Input:** $\alpha$, $\gamma$, $\varepsilon$, episodes $n$, target update frequency $C$

1. Initialize **online network** $Q(s,a;\theta)$ with random weights $\theta$

   Initialize **target network** $\hat{Q}(s,a;\theta^-)$ with $\theta^- \leftarrow \theta$ *(copy of online network)*

   Initialize step counter $t \leftarrow 0$

2. **For** episode $= 1, 2, \ldots, n$ *(outer loop — episodes)*:
   - Initialize $s$
   - **Repeat** for each step *(inner loop — time steps)*:
     - Choose $a$ using $\varepsilon$-greedy w.r.t. $Q(s,\cdot;\theta)$
     - Take action $a$, observe $r,  s'$
     - **Compute target using frozen target network** *($\theta^-$ is NOT updated here)*:
       $$y \leftarrow r + \gamma \max_{a'} \hat{Q}(s', a';  \theta^-)$$
     - **Update online network:**
       $$\mathcal{L}(\theta) = \bigl(y - Q(s,a;\theta)\bigr)^2, \qquad \theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$$
     - $t \leftarrow t + 1$; if $t \bmod C = 0$: $\theta^- \leftarrow \theta$ *(hard copy every $C$ steps)*
     - $s \leftarrow s'$
   - Until $s$ is terminal

3. Return $Q(\cdot,\cdot;\theta)$; derive $\pi(s) = \arg\max_a Q(s,a;\theta)$

✦ ✦ ✦

**Notes:**

- $\theta^-$ is **frozen** between updates $\Rightarrow$ target $y$ does not shift with every gradient step $\Rightarrow$ stabilizes learning.
- Alternative: **soft update** $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ with small $\tau$ (e.g., $\tau = 0.005$) every step.
- This alone still suffers from correlated samples $\Rightarrow$ combine with Experience Replay.

<a id='tricks-3-2'></a>

**Additional Known Engineering Tricks**

- <a id='trick-3-2-1'></a>**Polyak (soft) target update** — $\theta^-\leftarrow \tau\theta + (1-\tau)\theta^-$ every step with small $\tau$ (e.g., $0.005$). *Why it helps:* smoothly tracks the online network with controlled lag instead of taking a sudden jump every $C$ steps; the effective time constant is $1/\tau\approx 200$. *Interaction:* mathematically interpolates between hard-sync ($\tau=1$ every step) and frozen target ($\tau=0$); pick one of hard/soft, not both. Mandatory in DDPG/SAC where the actor's gradient flows through the target critic and sudden jumps would destabilise the actor.

- <a id='trick-3-2-2'></a>**Target network *only* for $\max_{a'} Q$, not for action selection at $s$** — keep $a\sim\varepsilon\text{-greedy}(Q(s,\cdot;\theta))$ from the *online* net. *Why it helps:* using $\theta^-$ for behaviour too would make exploration stale by $C$ steps; using $\theta$ for behaviour and $\theta^-$ only inside the regression target gives both fresh exploration and a stable regression problem.

- <a id='trick-3-2-3'></a>**Asymmetric $C$ scheduling** — start with small $C$ (≈100) during the noisy early phase and increase to large $C$ later. *Why it helps:* early on, the network is changing fast and a tight $C$ tracks reality; later, learning is finer and benefits from longer-frozen targets to reduce target jitter.

- **Decoupling the max** *(precursor to Double DQN)* — when the target network is in place, you have two networks lying around; use the online net to *select* the maximising action and the target net to *evaluate* it: $y = r + \gamma \hat{Q}(s', \arg\max_{a'} Q(s',a';\theta);\theta^-)$. *Why it helps:* same maximisation-bias fix as Double Q-learning, free from the existing two-network setup; ships as **Double DQN** in the full algorithm.

- <a id='trick-3-2-4'></a>**Target net for $V(s')$ in $n$-step bootstraps** — when extending to $n$-step DQN, evaluate the bootstrap leaf $Q(s_{t+n}, \cdot)$ with $\theta^-$ but compute the running sum of rewards with no network involvement. *Why it helps:* keeps the only network call in the target stable; the reward sum is data-only.

<a id='pseudo-3-2'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 3.2</strong> DQN with target network — Raw Pseudocode (without additional tricks) <em>[Mnih et al., 2015]</em>
</div>

Here is the pseudocode for the DQN with target network core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** $\alpha$, $\gamma$, $\varepsilon$, episodes $n$, target update frequency $C$  
Initialize online network $Q(s, a;  \theta)$  
Initialize target network $\hat{Q}(s, a;  \theta^{-})$ with $\theta^{-} \leftarrow \theta$  
Step counter $t \leftarrow 0$  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialize $s$  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;Choose $a$ from $s$ using $\varepsilon$-greedy w.r.t. $Q(s, \cdot;  \theta)$  
&emsp;&emsp;Take action $a$; observe $r,  s'$  
&emsp;&emsp;$y \leftarrow r + \gamma  \max_{a'} \hat{Q}(s', a';  \theta^{-})$ *(target from frozen net)*  
&emsp;&emsp;$\mathcal{L}(\theta) \leftarrow \bigl(  y - Q(s, a;  \theta)  \bigr)^{2}$  
&emsp;&emsp;$\theta \leftarrow \theta - \alpha  \nabla_{\theta}  \mathcal{L}(\theta)$  
&emsp;&emsp;$s \leftarrow s';\quad t \leftarrow t + 1$  
&emsp;&emsp;**if** $t \bmod C = 0$ **then** $\theta^{-} \leftarrow \theta$  
&emsp;**until** $s$ is terminal  
**end for**  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 3.2</strong> DQN with target network — Pseudocode with Additions (Engineering Tricks) <em>[Mnih et al., 2015]</em>
</div>

Here is the pseudocode for the DQN with target network algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** $\alpha$, $\gamma$, $\varepsilon$, episodes $n$, ✦ soft-update $\tau$ ✦ ([Polyak (soft) target update](#trick-3-2-1)), ✦ initial $C_0$, final $C_\text{final}$ ✦ ([Asymmetric $C$ scheduling](#trick-3-2-3)), ✦ bootstrap steps $m$ ✦ ([Target net for $V(s')$ in $n$-step bootstraps](#trick-3-2-5))  
Initialise online network $Q(s, a;  \theta)$  
Initialise target network $\hat{Q}(s, a;  \theta^{-})$ with $\theta^{-} \leftarrow \theta$  
Step counter $t \leftarrow 0$  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialise $s$  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;✦ Choose $a$ from $s$ using $\varepsilon$-greedy w.r.t. $Q(s, \cdot;  \theta)$ *(online net for behaviour, not target)* ✦ ([Target network *only* for $\max_{a'} Q$, not for action selection at $s$](#trick-3-2-2))  
&emsp;&emsp;Take action $a$; observe $r,  s'$  
&emsp;&emsp;✦ $a^* \leftarrow \arg\max_{a'} Q(s', a';  \theta)$ *(online net selects)* ✦ ([Decoupling the max](#trick-3-2-4))  
&emsp;&emsp;✦ $y \leftarrow r + \gamma  \hat{Q}(s', a^*;  \theta^{-})$ *(target net evaluates — Double DQN)* ✦ ([Decoupling the max](#trick-3-2-4))  
&emsp;&emsp;✦ *(For $m$-step bootstrap: $y \leftarrow \sum_{k=0}^{m-1}\gamma^k r_{t+k} + \gamma^m \hat{Q}(s_{t+m}, \cdot;  \theta^{-})$)* ✦ ([Target net for $V(s')$ in $n$-step bootstraps](#trick-3-2-5))  
&emsp;&emsp;$\mathcal{L}(\theta) \leftarrow \bigl(  y - Q(s, a;  \theta)  \bigr)^{2}$  
&emsp;&emsp;$\theta \leftarrow \theta - \alpha  \nabla_{\theta}  \mathcal{L}(\theta)$  
&emsp;&emsp;$s \leftarrow s';\quad t \leftarrow t + 1$  
&emsp;&emsp;✦ $\theta^{-} \leftarrow \tau \theta + (1 - \tau) \theta^{-}$ *(Polyak soft update every step)* ✦ ([Polyak (soft) target update](#trick-3-2-1))  
&emsp;&emsp;✦ *(Alternative: hard sync every $C$ steps, with $C$ increasing over training)* ✦ ([Asymmetric $C$ scheduling](#trick-3-2-3))  
&emsp;**until** $s$ is terminal  
**end for**  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='dqn-experience-replay'></a>

### DQN + Experience Replay (DQN+ER)

**Notation:**
- $\mathcal{D}$ — replay buffer storing past transitions
- $M$ — buffer capacity (max transitions stored; oldest discarded when $|\mathcal{D}| > M$)
- $B$ — minibatch size (transitions sampled per gradient update)
- $\text{done}$ — boolean terminal flag: $\text{true}$ iff $s'$ is a terminal state

Same as DQN+TN above:
- $\theta^-$ — frozen target network weights (a periodic hard copy of $\theta$; never updated by gradients)
- $\hat{Q}(s,a;\theta^-)$ — target network (separate from the online network $Q$)
- $C$ — target update frequency (gradient steps between hard copies $\theta^- \leftarrow \theta$)
- $t$ — global step counter

<a id='intro-3-3'></a>

**Introduction:** The full DQN (Mnih et al., 2015). Adds an ***experience replay*** buffer on top of the target network: every transition $(s,a,r,s',\text{done})$ is stored in a FIFO buffer $\mathcal{D}$, and minibatches are sampled uniformly at random for each gradient update. *Idea:* train on shuffled past transitions instead of consecutive ones, and reuse each transition many times. *Why it helps:* decorrelates samples (consecutive trajectories badly violate the i.i.d. assumption SGD relies on) and dramatically improves sample efficiency by recycling data.

**Algorithm:**

**Input:** $\alpha$, $\gamma$, $\varepsilon$, episodes $n$, buffer capacity $M$, minibatch size $B$, target update freq $C$

1. Initialize **online network** $Q(s,a;\theta)$; **target network** $\hat{Q}(s,a;\theta^-)$ with $\theta^- \leftarrow \theta$

   Initialize replay buffer $\mathcal{D}$ with capacity $M$; step counter $t \leftarrow 0$

2. **For** episode $= 1, 2, \ldots, n$ *(outer loop — episodes)*:
   - Initialize $s$
   - **Repeat** for each step *(inner loop — time steps)*:
     - Choose $a$ using $\varepsilon$-greedy w.r.t. $Q(s,\cdot;\theta)$
     - Take action $a$, observe $r,  s',  \text{done}$
     - Store $(s, a, r, s', \text{done})$ in $\mathcal{D}$ *(FIFO: oldest removed if $|\mathcal{D}| > M$)*
     - Sample $B$ transitions $\lbrace (s_j, a_j, r_j, s'_j, \text{done}_j)\rbrace$ uniformly from $\mathcal{D}$
     - **Compute targets** for each $j$ in minibatch:
       $$y_j = \begin{cases} r_j & \text{if } \text{done}_j \\ r_j + \gamma \max_{a'} \hat{Q}(s'_j, a';  \theta^-) & \text{otherwise} \end{cases}$$
     - **Gradient update:**
       $$\mathcal{L}(\theta) = \frac{1}{B}\sum_j \bigl(y_j - Q(s_j, a_j;\theta)\bigr)^2, \qquad \theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$$
     - $t \leftarrow t+1$; if $t \bmod C = 0$: $\theta^- \leftarrow \theta$
     - $s \leftarrow s'$
   - Until $s$ is terminal ($\text{done} = \text{true}$)

3. Return $Q(\cdot,\cdot;\theta)$; derive $\pi(s) = \arg\max_a Q(s,a;\theta)$

✦ ✦ ✦

**Notes:**

- **Experience Replay** breaks temporal correlation: the minibatch contains transitions from different episodes and time steps $\Rightarrow$ approximately i.i.d. samples for SGD.
- Each transition can be reused many times $\Rightarrow$ better sample efficiency.
- $\mathcal{D}$ is a fixed-size FIFO queue (e.g., $M = 10^6$).
- $\varepsilon$ is often annealed: start at $1.0$ (pure exploration) $\to$ decay to $0.01$ over training.
- This is the standard **full DQN** algorithm. In practice, both tricks are always used together.

<a id='tricks-3-3'></a>

**Additional Known Engineering Tricks**

- **Prioritized Experience Replay (PER)** *(Schaul et al., 2016)* — sample transition $j$ with probability $\propto |\delta_j|^\alpha$ (the magnitude of its last TD error); correct the bias with importance-sampling weights $w_j = (1/(N P_j))^\beta$. *Why it helps:* high-error transitions carry more learning signal, so prioritising them speeds up training $\sim 2\times$ on Atari. *Interaction:* the IS weight is essential — without it the $Q$ estimate becomes biased; sum-trees give $O(\log N)$ sampling.

- **Double DQN** *(van Hasselt, Guez & Silver, 2016)* — change the target to $y_j = r_j + \gamma \hat{Q}(s'_j, \arg\max_{a'} Q(s'_j,a';\theta); \theta^-)$ — online net selects, target net evaluates. *Why it helps:* removes the systematic overestimation bias of $\max_{a'} \hat{Q}$ for free (no extra parameters). *Interaction:* needs a target network already, so combines trivially with the trick from the previous section.

- **Dueling network architecture** *(Wang et al., 2016)* — split the head into a state-value stream $V(s)$ and an advantage stream $A(s,a)$, combine as $Q(s,a) = V(s) + (A(s,a) - \tfrac{1}{|A|}\sum_{a'} A(s,a'))$. *Why it helps:* lets the network learn the *value of being in a state* without committing to any particular action — important when actions don't affect the immediate value (e.g., navigating an empty corridor).

- **Noisy Networks for exploration** *(Fortunato et al., 2018)* — replace the linear layers in the Q-network with $y = (\mu_w + \sigma_w \odot \varepsilon_w)x + (\mu_b + \sigma_b \odot \varepsilon_b)$, with $\varepsilon$ resampled each forward pass. *Why it helps:* learnable *parameter* noise replaces $\varepsilon$-greedy; exploration becomes state-conditional and self-tuning. *Interaction:* makes $\varepsilon$-greedy unnecessary and is required for Rainbow.

- <a id='trick-3-3-1'></a>**n-step bootstrap returns in the buffer** — store transitions as $(s_t, a_t, R^{(n)}_t, s_{t+n})$ with $R^{(n)}_t=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}$. *Why it helps:* faster reward propagation, same trade-off as $n$-step TD; typical $n=3$ on Atari. *Interaction:* technically makes the target *slightly* off-policy when combined with $\varepsilon$-greedy, but in practice the small bias is dominated by faster convergence.

- **Distributional RL — C51 / QR-DQN / IQN** *(Bellemare et al., 2017; Dabney et al., 2018a,b)* — replace the scalar $Q(s,a)$ with a *distribution* over returns (51-bin categorical for C51; quantile regression for QR-DQN; implicit quantile for IQN). *Why it helps:* the network learns a richer signal (variance, skew, multimodality of returns), which empirically yields large gains; the policy is still $\arg\max_a \mathbb{E}[Z(s,a)]$.

- **Rainbow** *(Hessel et al., 2018)* — combines all six tricks above (PER + Double + Dueling + Noisy + n-step + Distributional) plus the basic target net + replay. *Why it helps:* every component fixes a different failure mode of naive DQN; ablation studies show they are largely complementary.

- <a id='trick-3-3-2'></a>**Replay warm-up / minimum buffer fill** — collect $\sim50\text{k}$ random-policy transitions before any gradient update. *Why it helps:* ensures the very first minibatches are diverse enough not to overfit on a near-empty buffer; standard $|\mathcal{D}|=10^6$ FIFO with min-size $\sim 5$% of capacity.

<a id='pseudo-3-3'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 3.3</strong> Deep Q-learning with experience replay (full DQN) — Raw Pseudocode (without additional tricks) <em>[Mnih et al., 2015]</em>
</div>

Here is the pseudocode for the DQN with experience replay (full DQN) core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** $\alpha$, $\gamma$, $\varepsilon$, episodes $n$, buffer capacity $M$, minibatch size $B$, target update freq $C$  
Initialize online network $Q(s, a;  \theta)$; target network $\hat{Q}(s, a;  \theta^{-})$ with $\theta^{-} \leftarrow \theta$  
Initialize replay buffer $\mathcal{D}$ with capacity $M$; step counter $t \leftarrow 0$  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialize $s$  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;Choose $a$ from $s$ using $\varepsilon$-greedy w.r.t. $Q(s, \cdot;  \theta)$  
&emsp;&emsp;Take action $a$; observe $r,  s',  \text{done}$  
&emsp;&emsp;Store transition $(s, a, r, s', \text{done})$ in $\mathcal{D}$  
&emsp;&emsp;Sample minibatch $\lbrace (s_j, a_j, r_j, s'_j, \text{done}_j)\rbrace _{j=1}^{B} \sim \mathcal{D}$  
&emsp;&emsp;**for each** $j$ **do**  
&emsp;&emsp;&emsp;$y_j \leftarrow r_j + \gamma (1 - \text{done}_j)  \max_{a'} \hat{Q}(s'_j, a';  \theta^{-})$  
&emsp;&emsp;**end for**  
&emsp;&emsp;$\mathcal{L}(\theta) \leftarrow \dfrac{1}{B}  \sum_{j=1}^{B} \bigl(  y_j - Q(s_j, a_j;  \theta)  \bigr)^{2}$  
&emsp;&emsp;$\theta \leftarrow \theta - \alpha  \nabla_{\theta}  \mathcal{L}(\theta)$  
&emsp;&emsp;$s \leftarrow s';\quad t \leftarrow t + 1$  
&emsp;&emsp;**if** $t \bmod C = 0$ **then** $\theta^{-} \leftarrow \theta$  
&emsp;**until** $\text{done} = \mathbf{true}$  
**end for**  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 3.3</strong> Deep Q-learning with experience replay (full DQN) — Pseudocode with Additions (Engineering Tricks) <em>[Mnih et al., 2015]</em>
</div>

Here is the pseudocode for the DQN with experience replay (full DQN) algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** $\alpha$, $\gamma$, episodes $n$, buffer capacity $M$, minibatch $B$, target update freq $C$, ✦ PER exponent $\alpha_\text{per}$, IS exponent $\beta_\text{per}$ ✦ ([Prioritized Experience Replay (PER)](#trick-3-3-1)), ✦ bootstrap steps $m$ ✦ ([n-step bootstrap returns in the buffer](#trick-3-3-5)), ✦ warm-up steps $W$ ✦ ([Replay warm-up / minimum buffer fill](#trick-3-3-8))  
✦ Initialise $Q$-network with dueling architecture: $Q(s,a) = V(s;\theta_V) + \bigl(A(s,a;\theta_A) - \tfrac{1}{|A|}\sum_{a'}A(s,a';\theta_A)\bigr)$ ✦ ([Dueling network architecture](#trick-3-3-3))  
✦ Replace linear layers with noisy layers: $y = (\mu_w + \sigma_w \odot \varepsilon)x + (\mu_b + \sigma_b \odot \varepsilon_b)$ ✦ ([Noisy Networks for exploration](#trick-3-3-4))  
Initialise target $\hat{Q}$ with $\theta^{-} \leftarrow \theta$  
Initialise prioritised replay buffer $\mathcal{D}$ with capacity $M$ (sum-tree)  
Step counter $t \leftarrow 0$  
✦ **for** $t = 1, \ldots, W$ **do** collect random-policy transitions into $\mathcal{D}$ **end for** ✦ ([Replay warm-up / minimum buffer fill](#trick-3-3-8))  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialise $s$  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;✦ $a \leftarrow \arg\max_a Q(s, a;  \theta)$ *($\varepsilon$-greedy unnecessary with noisy nets)* ✦ ([Noisy Networks for exploration](#trick-3-3-4))  
&emsp;&emsp;Take action $a$; observe $r,  s',  \text{done}$  
&emsp;&emsp;✦ Store $m$-step transition $(s_t, a_t, R^{(m)}_t, s_{t+m})$ with $R^{(m)}=\sum_{k=0}^{m-1}\gamma^k r_{t+k+1}$ ✦ ([n-step bootstrap returns in the buffer](#trick-3-3-5))  
&emsp;&emsp;✦ Sample minibatch $j \sim P(j) \propto |\delta_j|^{\alpha_\text{per}}$ from $\mathcal{D}$; compute IS weights $w_j = (N P_j)^{-\beta_\text{per}}$ ✦ ([Prioritized Experience Replay (PER)](#trick-3-3-1))  
&emsp;&emsp;**for each** $j$ **do**  
&emsp;&emsp;&emsp;✦ $a^*_j \leftarrow \arg\max_{a'} Q(s'_j, a';  \theta)$; $y_j \leftarrow R^{(m)}_j + \gamma^m(1-\text{done}_j) \hat{Q}(s'_j, a^*_j;  \theta^{-})$ *(Double DQN + $m$-step)* ✦ ([Double DQN](#trick-3-3-2) / [n-step bootstrap returns in the buffer](#trick-3-3-5))  
&emsp;&emsp;**end for**  
&emsp;&emsp;✦ $\mathcal{L}(\theta) \leftarrow \frac{1}{B}\sum_j w_j\bigl(  y_j - Q(s_j, a_j;  \theta)  \bigr)^{2}$ *(IS-weighted loss)* ✦ ([Prioritized Experience Replay (PER)](#trick-3-3-1))  
&emsp;&emsp;✦ Update priorities $|\delta_j|$ in $\mathcal{D}$ ✦ ([Prioritized Experience Replay (PER)](#trick-3-3-1))  
&emsp;&emsp;✦ *(Full Rainbow: all above + distributional C51/QR-DQN head replacing scalar $Q$)* ✦ ([Distributional RL — C51 / QR-DQN / IQN](#trick-3-3-6) / [Rainbow](#trick-3-3-7))  
&emsp;&emsp;$\theta \leftarrow \theta - \alpha  \nabla_{\theta}  \mathcal{L}(\theta);\quad s \leftarrow s';\quad t \leftarrow t + 1$  
&emsp;&emsp;**if** $t \bmod C = 0$ **then** $\theta^{-} \leftarrow \theta$  
&emsp;**until** $\text{done} = \mathbf{true}$  
**end for**  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

✦ ✦ ✦
## Deep RL — Policy Gradient Methods
✦ ✦ ✦

<a id='vanilla-pg-loss'></a>

### What Is the Vanilla Policy Gradient Loss?

All policy gradient methods share a common ancestor: the **policy gradient theorem**, which gives the gradient of the expected return $J(\theta) = \mathbb{E}_{\pi_\theta}[R]$ with respect to the policy parameters:

$$\nabla_\theta J(\theta)  =  \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T}\nabla_\theta \log\pi_\theta(a_t \mid s_t) \Psi_t\right]$$

The **vanilla policy gradient loss** is the objective whose gradient recovers this formula:

$$\boxed{ \mathcal{L}^{\text{VPG}}(\theta)  =  - \mathbb{E}\left[\sum_{t}\log\pi_\theta(a_t \mid s_t) \Psi_t\right] }$$

where $\Psi_t$ is a scalar **signal** that tells the gradient *how good* action $a_t$ was. Different choices of $\Psi_t$ yield different algorithms while preserving the same loss structure:

| $\Psi_t$ | Algorithm |
|---|---|
| $G_t$ (full Monte-Carlo return) | REINFORCE |
| $G_t - b(s_t)$ (baseline-subtracted return) | REINFORCE with baseline |
| $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ (TD error) | Actor-Critic |
| $A_t = Q(s_t, a_t) - V(s_t)$ (advantage) | A2C / A3C |

**Intuition:** increase the log-probability of actions that led to above-average outcomes; decrease it for below-average ones. The vanilla loss implements this idea in its simplest form — no surrogate objectives, no clipping, no entropy augmentation, no importance-sampling correction.

Methods that **depart** from this vanilla template each replace or augment the objective in a fundamental way: PPO introduces a clipped importance-sampling surrogate, SAC adds a maximum-entropy bonus to the reward, DDPG bypasses stochastic log-probabilities entirely via the deterministic policy gradient, and CQL adds a conservative regulariser for offline learning.

<div style="margin:28px 0 14px 0; padding:4px 0; border-bottom:1.5px solid #888; font-family:'Latin Modern Roman','Times New Roman',serif;">
<span style="font-size:1.15em; font-weight:800; font-style:italic;">✦&ensp;Vanilla Policy Gradient</span>
</div>

<a id='reinforce'></a>

### REINFORCE (Monte Carlo Policy Gradient)

*Model-free, on-policy, Monte Carlo policy gradient; learns from* ***complete*** *episodes.*
*Instead of learning $Q/V$, directly parameterize and optimize the* ***policy*** $\pi(a \mid s;\theta)$*.*

**Notation:**
- $\theta$ — policy network weights (*not* a convergence threshold; learnable parameters of $\pi$)
- $J(\theta)$ — performance objective: expected total discounted return under policy $\pi(\cdot;\theta)$
- $G_t$ — discounted return from step $t$ to episode end: $G_t = \sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}$
- $T$ — terminal time step of the episode
- $b(s_t)$ — baseline (any function of $s_t$, e.g. $V(s_t)$); subtracted to reduce gradient variance

<a id='intro-4-1'></a>

**Introduction:** Model-free, on-policy ***Monte Carlo policy gradient*** — directly parameterises and optimises the policy $\pi(a \mid s;\theta)$ rather than a value function. Learns from ***complete*** episodes. *Idea:* the gradient of expected return is $\nabla J(\theta) = \mathbb{E}\left[\sum_t G_t  \nabla\log\pi(a_t \mid s_t;\theta)\right]$ — increase the log-probability of actions weighted by the return that followed them. *Why it helps:* works natively for continuous and stochastic policies, where $\max_a Q$ is intractable; learns stochastic policies in their own right. *Cost:* high variance — typically reduced with a learned baseline (see Actor–Critic).

**Algorithm:**

**Input:** policy network $\pi(a \mid s;\theta)$, learning rate $\alpha$, discount $\gamma$, episodes $n$

1. Initialize policy network $\pi(a \mid s;\theta)$ with random weights $\theta$

2. **For** episode $= 1, 2, \ldots, n$ *(outer loop — episodes)*:

   - **— Generate full episode using current $\pi$ —**

   $$s_0,  a_0,  r_1,  s_1,  a_1,  r_2,  \ldots,  s_{T-1},  a_{T-1},  r_T \quad \text{(run until terminal)}$$

   - **— Compute returns —**

   - For $t = 0, 1, \ldots, T-1$:
     $$G_t = \sum_{k=0}^{T-t-1} \gamma^k  r_{t+k+1}$$

   - **— Policy gradient update —**

   $$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \gamma^t  G_t  \nabla_\theta \log \pi(a_t \mid s_t;\theta)$$

3. Return $\pi(\cdot \mid \cdot;\theta)$

✦ ✦ ✦

**Policy gradient theorem:**

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[\sum_t \gamma^t  G_t  \nabla_\theta \log \pi(a_t \mid s_t;\theta)\right]$$

**Intuition:**

- $G_t$ high $\Rightarrow$ increase $\log\pi(a_t \mid s_t;\theta)$ $\Rightarrow$ make $a_t$ **more likely**
- $G_t$ low $\Rightarrow$ decrease $\log\pi(a_t \mid s_t;\theta)$ $\Rightarrow$ make $a_t$ **less likely**

**Notes:**

- Must wait for a **complete** episode before updating (like MC) — cannot learn online.
- **High variance:** $G_t$ can vary wildly across episodes $\Rightarrow$ noisy gradient estimates.
- Common fix — subtract a **baseline** $b(s_t)$ from $G_t$ (reduces variance without adding bias):
  $$\theta \leftarrow \theta + \alpha \sum_t \gamma^t \bigl(G_t - b(s_t)\bigr) \nabla_\theta \log \pi(a_t \mid s_t;\theta)$$
  A common baseline is $V(s_t)$ $\Rightarrow$ this leads to **Actor-Critic** methods (see below).
- Outputs a **stochastic** policy $\pi(a \mid s;\theta)$, unlike Q-learning which outputs a deterministic policy.

<a id='tricks-4-1'></a>

**Additional Known Engineering Tricks**

- <a id='trick-4-1-1'></a>**Baseline subtraction** — replace $G_t$ in the gradient with $G_t - b(s_t)$ where $b$ is any function of state (commonly $b(s)=V(s)$ from a learned critic). *Why it helps:* $\mathbb{E}_a[b(s)\nabla\log\pi(a\mid s)] = 0$, so subtracting a baseline reduces variance *without adding bias* — by far the most impactful single trick for REINFORCE. *Interaction:* learning $b=V$ alongside $\pi$ gives Actor-Critic — see Algorithm 4.2.

- <a id='trick-4-1-2'></a>**Return standardisation (whitening)** — within each episode (or batch of episodes), shift and scale: $G_t \leftarrow (G_t - \mu_G) / (\sigma_G + 10^{-8})$. *Why it helps:* removes the dependence on absolute reward scale and stabilises the gradient magnitude for Adam/SGD; the learning rate becomes scale-invariant. *Interaction:* using *both* whitening and a learned baseline subtracts the mean twice — usually pick one; whitening is the cheap drop-in if no critic is trained.

- <a id='trick-4-1-3'></a>**Adam with low learning rate ($10^{-4} \text{–} 10^{-3}$)** — REINFORCE's gradient variance is enormous, so a small adaptive step size is essential. *Why it helps:* Adam's per-parameter normalisation hides much of the gradient variance from the optimiser, allowing the small effective step to still make progress.

- <a id='trick-4-1-4'></a>**Global-norm gradient clipping** — if $\Vert \nabla\theta\Vert  > c$, scale to norm $c$ (typical $c\in[0.5, 5]$). *Why it helps:* a single high-return outlier episode can blow up the gradient; clipping caps the damage one bad batch can do.

<a id='pseudo-4-1'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.1</strong> REINFORCE (Monte Carlo policy gradient) — Raw Pseudocode (without additional tricks) <em>[Williams, 1992; Sutton & Barto, §13.3]</em>
</div>

Here is the pseudocode for the REINFORCE core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** policy network $\pi(a \mid s;  \theta)$, learning rate $\alpha$, discount $\gamma$, episodes $n$  
Initialize $\theta$ randomly  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Generate an episode using $\pi_\theta$: $s_0, a_0, r_1,  s_1, a_1, r_2,  \ldots,  s_{T-1}, a_{T-1}, r_T$  
&emsp;**for** $t = 0, 1, \ldots, T - 1$ **do**  
&emsp;&emsp;$G_t \leftarrow \displaystyle\sum_{k = t + 1}^{T} \gamma^{ k - t - 1}  r_k$  
&emsp;&emsp;$\theta \leftarrow \theta + \alpha  \gamma^{ t}  G_t  \nabla_{\theta} \log \pi(a_t \mid s_t;  \theta)$  
&emsp;**end for**  
**end for**  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.1</strong> REINFORCE (Monte Carlo policy gradient) — Pseudocode with Additions (Engineering Tricks) <em>[Williams, 1992; Sutton & Barto, §13.3]</em>
</div>

Here is the pseudocode for the REINFORCE algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** policy network $\pi(a \mid s;  \theta)$, ✦ Adam optimiser with $\alpha \in [10^{-4},  10^{-3}]$ ✦ ([Adam with low learning rate ($10^{-4} \text{–} 10^{-3}$)](#trick-4-1-3)), discount $\gamma$, episodes $n$, ✦ gradient clip norm $c$ ✦ ([Global-norm gradient clipping](#trick-4-1-4))  
Initialise $\theta$ randomly  
✦ *(Optionally initialise a baseline $b(s)$ — e.g., a learned $V(s;w)$)* ✦ ([Baseline subtraction](#trick-4-1-1))  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Generate an episode using $\pi_\theta$: $s_0, a_0, r_1,  s_1, a_1, r_2,  \ldots,  s_{T-1}, a_{T-1}, r_T$  
&emsp;**for** $t = 0, 1, \ldots, T - 1$ **do**  
&emsp;&emsp;$G_t \leftarrow \displaystyle\sum_{k = t + 1}^{T} \gamma^{ k - t - 1}  r_k$  
&emsp;**end for**  
&emsp;✦ Standardise returns: $G_t \leftarrow (G_t - \mu_G)/(\sigma_G + 10^{-8})$ over $\lbrace G_0, \ldots, G_{T-1}\rbrace$ ✦ ([Return standardisation (whitening)](#trick-4-1-2))  
&emsp;**for** $t = 0, 1, \ldots, T - 1$ **do**  
&emsp;&emsp;✦ $\theta \leftarrow \theta + \alpha  \gamma^{ t} (G_t - b(s_t))  \nabla_{\theta} \log \pi(a_t \mid s_t;  \theta)$ *(baseline-subtracted gradient)* ✦ ([Baseline subtraction](#trick-4-1-1))  
&emsp;**end for**  
&emsp;✦ **if** $\lVert\nabla\theta\rVert > c$ **then** $\nabla\theta \leftarrow c \cdot \nabla\theta / \lVert\nabla\theta\rVert$ ✦ ([Global-norm gradient clipping](#trick-4-1-4))  
&emsp;✦ Apply Adam step ✦ ([Adam with low learning rate ($10^{-4} \text{–} 10^{-3}$)](#trick-4-1-3))  
**end for**  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='actor-critic'></a>

### Actor-Critic (AC)

**Notation:**
- $w$ — critic (value network) weights
- $\alpha_\theta$ — actor learning rate
- $\alpha_w$ — critic learning rate
- $G_t$ — discounted return from step $t$ (same as REINFORCE)
- $\gamma^t$ — discounting applied to the policy gradient at time $t$

<a id='intro-4-2'></a>

**Introduction:** Combines a policy gradient (***Actor***, $\pi(a\mid s;\theta)$) with a learned value function (***Critic***, $V(s;w)$). The Critic's TD error $\delta = r + \gamma V(s';w) - V(s;w)$ replaces the Monte Carlo return $G_t$ as the policy-gradient signal.

- **Actor:** policy network $\pi(a \mid s;\theta)$ — decides which action to take
- **Critic:** value network $V(s;w)$ — evaluates how good the current state is

*Idea:* let the Actor choose actions and the Critic score the resulting states; use the Critic's verdict to push the Actor's log-probabilities. *Why it helps over REINFORCE:* much lower variance (TD error / advantage instead of full return), and updates can happen ***online*** at every step — no need to wait for episode termination.

**Algorithm:**

**Input:** actor rate $\alpha_\theta$, critic rate $\alpha_w$, discount $\gamma$, episodes $n$

1. Initialize Actor $\pi(a \mid s;\theta)$ and Critic $V(s;w)$ with random weights

2. **For** episode $= 1, 2, \ldots, n$ *(outer loop — episodes)*:
   - Initialize $s$
   - **Repeat** for each step *(inner loop — time steps)*:
     - $a \sim \pi(\cdot \mid s;\theta)$ *(Actor chooses action)*
     - Take action $a$, observe $r,  s'$
     - **TD error (Critic's evaluation):**
       $$\delta \leftarrow r + \gamma V(s';w) - V(s;w)$$
       *($\delta > 0$: outcome was better than expected; $\delta < 0$: worse than expected)*
     - **Update Critic** *(minimize TD error)*:
       $$w \leftarrow w + \alpha_w  \delta  \nabla_w V(s;w)$$
     - **Update Actor** *(policy gradient with $\delta$ as advantage)*:
       $$\theta \leftarrow \theta + \alpha_\theta  \gamma^t  \delta  \nabla_\theta \log \pi(a \mid s;\theta)$$
       *($\delta$ replaces $G_t$ from REINFORCE)*
     - $s \leftarrow s'$
   - Until $s$ is terminal

3. Return $\pi(\cdot \mid \cdot;\theta)$

✦ ✦ ✦

**Notes:**

- Can update **every step** (online), not just at episode end.
- The TD error $\delta = r + \gamma V(s') - V(s)$ estimates the **advantage** $A(s,a)$ = "how much better was this action than what I expected from this state?"
- **Lower variance** than REINFORCE ($\delta$ is less noisy than $G_t$), but introduces **bias** (because $V(s';w)$ is an approximation).
- Two separate networks: Actor ($\theta$) and Critic ($w$), each with its own learning rate. They can optionally share lower layers for feature extraction.

<a id='tricks-4-2'></a>

**Additional Known Engineering Tricks**

- <a id='trick-4-2-1'></a>**Shared trunk + two heads** — let actor and critic share the bulk of the network (conv stack for images, MLP body for state vectors), branching only at the last layer into a policy head and a value head. *Why it helps:* one feature extractor is forced to encode information useful for both objectives, which acts as a regulariser; halves parameter count and forward-pass cost. *Interaction:* if the actor and critic learning rates differ by orders of magnitude, the shared trunk's gradient can be dominated by one head — fix with separate optimisers per head or per-head loss-scale weights.

- <a id='trick-4-2-2'></a>**Target value network for the critic** — keep a slow-moving copy $V(s;w^-)$ and use $\delta = r + \gamma V(s';w^-) - V(s;w)$ instead of $V(s';w)$. *Why it helps:* same logic as DQN+TN — prevents the critic's bootstrap target from chasing its own moving estimate; particularly important when the actor and critic share weights, because actor gradients perturb the critic indirectly.

- <a id='trick-4-2-3'></a>**$n$-step / multi-step bootstrapped returns** — replace the one-step $\delta$ with $\delta^{(n)} = \sum_{k=0}^{n-1}\gamma^k r_{t+k+1} + \gamma^n V(s_{t+n};w) - V(s_t;w)$. *Why it helps:* the same bias-variance dial as $n$-step TD, but inside the actor's advantage signal; recovers the full episode return at $n=T$ (REINFORCE-style) and one-step TD at $n=1$.

- **Entropy regularisation** $\beta H(\pi(\cdot\mid s))$ — add a small positive bonus on policy entropy to the actor's objective. *Why it helps:* prevents premature collapse of the stochastic policy onto a single action; on small problems the basic AC can converge to a near-deterministic suboptimal policy quickly without it. *Interaction:* this is what A2C/A3C and PPO inherit as a non-negotiable component.

- <a id='trick-4-2-4'></a>**Eligibility traces — AC($\lambda$)** — maintain traces $e_\theta, e_w$ on both networks and update with $\delta$ scaled by the trace. *Why it helps:* offline-equivalent of the $\lambda$-return for both actor and critic, with $O(1)$ amortised cost per step.

- **Pop-Art / adaptive reward normalisation** *(van Hasselt et al., 2016)* — predict normalised values $\tilde V$ and adapt the output layer's affine to preserve $V$'s scale as the running stats change. *Why it helps:* lets the critic learn across tasks with wildly different reward scales without retuning the learning rate; the actor sees a stable advantage signal regardless.

<a id='pseudo-4-2'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.2</strong> One-step Actor–Critic — Raw Pseudocode (without additional tricks) <em>[Sutton & Barto, §13.5]</em>
</div>

Here is the pseudocode for the Actor–Critic core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** actor lr $\alpha_\theta$, critic lr $\alpha_w$, discount $\gamma$, episodes $n$  
Initialize policy parameters $\theta$ and value parameters $w$ randomly  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialize $s$  
&emsp;$I \leftarrow 1$  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;Sample $a \sim \pi(\cdot \mid s;  \theta)$  
&emsp;&emsp;Take action $a$; observe $r,  s'$  
&emsp;&emsp;$\delta \leftarrow r + \gamma  V(s';  w) - V(s;  w)$ *(TD error / advantage)*  
&emsp;&emsp;$w \leftarrow w + \alpha_w  \delta  \nabla_{w}  V(s;  w)$  
&emsp;&emsp;$\theta \leftarrow \theta + \alpha_\theta  I  \delta  \nabla_{\theta} \log \pi(a \mid s;  \theta)$  
&emsp;&emsp;$I \leftarrow \gamma  I$  
&emsp;&emsp;$s \leftarrow s'$  
&emsp;**until** $s$ is terminal  
**end for**  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.2</strong> One-step Actor–Critic — Pseudocode with Additions (Engineering Tricks) <em>[Sutton & Barto, §13.5]</em>
</div>

Here is the pseudocode for the Actor–Critic algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** actor lr $\alpha_\theta$, critic lr $\alpha_w$, discount $\gamma$, ✦ trace decay $\lambda$ ✦ ([Eligibility traces — AC($\lambda$)](#trick-4-2-5)), ✦ entropy coefficient $\beta$ ✦ ([Entropy regularisation](#trick-4-2-4)), ✦ bootstrap steps $m$ ✦ ([n-step / multi-step bootstrapped returns](#trick-4-2-3)), episodes $n$  
✦ Initialise shared-trunk network with actor head $\pi(\cdot\mid s;\theta)$ and critic head $V(s;w)$ ✦ ([Shared trunk + two heads](#trick-4-2-1))  
✦ Initialise target critic $V(s;w^-)$ with $w^- \leftarrow w$ ✦ ([Target value network for the critic](#trick-4-2-2))  
✦ Initialise Pop-Art running stats $(\mu_r,\sigma_r)$ ✦ ([Pop-Art / adaptive reward normalisation](#trick-4-2-6))  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialise $s$; $I \leftarrow 1$  
&emsp;✦ $e_\theta \leftarrow 0, e_w \leftarrow 0$ *(eligibility traces)* ✦ ([Eligibility traces — AC($\lambda$)](#trick-4-2-5))  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;Sample $a \sim \pi(\cdot \mid s;  \theta)$  
&emsp;&emsp;Take action $a$; observe $r,  s'$  
&emsp;&emsp;✦ $r \leftarrow (r - \mu_r)/\sigma_r$; update running stats *(Pop-Art)* ✦ ([Pop-Art / adaptive reward normalisation](#trick-4-2-6))  
&emsp;&emsp;✦ $\delta^{(m)} \leftarrow \sum_{k=0}^{m-1}\gamma^k r_{t+k+1} + \gamma^m V(s_{t+m};  w^-) - V(s_t;  w)$ *($m$-step TD error via target critic)* ✦ ([n-step / multi-step bootstrapped returns](#trick-4-2-3) / [Target value network for the critic](#trick-4-2-2))  
&emsp;&emsp;✦ $e_w \leftarrow \gamma\lambda e_w + \nabla_w V(s;w)$; $w \leftarrow w + \alpha_w \delta^{(m)} e_w$ ✦ ([Eligibility traces — AC($\lambda$)](#trick-4-2-5))  
&emsp;&emsp;✦ $e_\theta \leftarrow \gamma\lambda e_\theta + I \nabla_\theta\log\pi(a\mid s;\theta)$ ✦ ([Eligibility traces — AC($\lambda$)](#trick-4-2-5))  
&emsp;&emsp;✦ $\theta \leftarrow \theta + \alpha_\theta \delta^{(m)} e_\theta + \beta \nabla_\theta H\bigl(\pi(\cdot\mid s)\bigr)$ *(+ entropy bonus)* ✦ ([Eligibility traces — AC($\lambda$)](#trick-4-2-5) / [Entropy regularisation](#trick-4-2-4))  
&emsp;&emsp;$I \leftarrow \gamma  I$; $s \leftarrow s'$  
&emsp;**until** $s$ is terminal  
**end for**  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='a2c'></a>

### A2C (Advantage Actor-Critic)

**Notation:**
- $K$ — number of parallel environment workers
- $n$ — rollout length (steps each worker collects before a gradient update)
- $\beta$ — entropy regularisation coefficient
- $H(\pi(\cdot \mid s;\theta))$ — policy entropy at state $s$: ${-\sum_a \pi(a \mid s)\log\pi(a \mid s)}$; higher $\Rightarrow$ more uniform distribution $\Rightarrow$ more exploration
- $\mathrm{Adv}_i^k$ — advantage estimate for worker $k$ at step $i$: $R - V(s_i^k;w)$

Same as Actor-Critic above:
- $w$ — critic (value network) weights
- $\alpha_\theta$ — actor learning rate
- $\alpha_w$ — critic learning rate

<a id='intro-4-3'></a>

**Introduction:** Synchronous advantage actor–critic. Runs $K$ parallel environment workers under a shared Actor and Critic; ***all workers finish their $n$-step rollouts*** before a single combined gradient update is applied. *Idea:* use the $n$-step advantage $\mathrm{Adv}_i = R_i - V(s_i;w)$ (lower variance than the MC return, less biased than the 1-step TD) to drive the policy gradient, and rely on parallel rollouts in different states for decorrelation. *Why it helps:* batches of decorrelated experiences stabilise training (no replay buffer needed) and parallelism keeps GPU utilisation high; in practice often matches A3C while being simpler and avoiding stale-gradient issues.

**Algorithm:**

**Input:** $\alpha_\theta$, $\alpha_w$, $\gamma$, rollout length $n$, parallel workers $K$, entropy coefficient $\beta$

1. Initialize shared Actor $\pi(a \mid s;\theta)$ and shared Critic $V(s;w)$

   Launch $K$ parallel environment copies: $\text{env}_1, \text{env}_2, \ldots, \text{env}_K$

2. **Repeat** until convergence *(outer loop — updates)*:

   - **— Each worker $k$ collects $n$ steps in parallel (synchronously) —**

   - For each worker $k = 1, \ldots, K$ and step $i = 1, \ldots, n$:
     - $a_i^k \sim \pi(\cdot \mid s_i^k;\theta)$; take $a_i^k$ in $\text{env}_k$, observe $r_i^k,  s_i^{\prime k}$

   - **— Compute $n$-step returns and advantages —**

   - For each worker $k$:
     $$R \leftarrow \begin{cases} 0 & \text{if } s_n^k \text{ is terminal} \\ V(s_n^k;w) & \text{otherwise (bootstrap from Critic)} \end{cases}$$
     - For $i = n{-}1, \ldots, 0$ (backwards): $\quad R \leftarrow r_{i+1}^k + \gamma R$; $\quad \mathrm{Adv}_i^k \leftarrow R - V(s_i^k;w)$

   - **— Aggregate losses across all $K$ workers —**

   $$\mathcal{L}_\text{actor}(\theta) = -\frac{1}{K}\sum_k\sum_i \mathrm{Adv}_i^k \cdot \log\pi(a_i^k \mid s_i^k;\theta)  -  \beta  H\bigl(\pi(\cdot \mid s_i^k;\theta)\bigr)$$

   $$\mathcal{L}_\text{critic}(w) = \frac{1}{K}\sum_k\sum_i \bigl(R - V(s_i^k;w)\bigr)^2$$

   - **— Single synchronous gradient update —**

   $$\theta \leftarrow \theta - \alpha_\theta \nabla_\theta \mathcal{L}_\text{actor}, \qquad w \leftarrow w - \alpha_w \nabla_w \mathcal{L}_\text{critic}$$

3. Return $\pi(\cdot \mid \cdot;\theta)$

✦ ✦ ✦

**Notes:**

- **Synchronous:** all $K$ workers finish their rollouts, then ONE gradient update happens. Simpler and more stable than A3C. No stale gradients.
- **Entropy bonus** $\beta H(\pi)$ encourages exploration by penalizing overly deterministic policies: $H(\pi(\cdot \mid s)) = -\sum_a \pi(a \mid s)\log\pi(a \mid s)$. Higher entropy $\Rightarrow$ more uniform $\Rightarrow$ more exploration.
- $K$ parallel envs $\Rightarrow$ each update uses $K \times n$ transitions $\Rightarrow$ lower-variance gradient estimates.
- $n$-step returns combine TD bootstrapping with multi-step actual rewards (like $n$-step TD).
- In practice, actor and critic often share a neural network body with two output heads.

<a id='tricks-4-3'></a>

**Additional Known Engineering Tricks**

- **Generalised Advantage Estimation — GAE($\lambda$)** *(Schulman et al., 2016)* — replace $R - V(s)$ with $\hat A_t = \sum_{l\geq 0}(\gamma\lambda)^l \delta_{t+l}$. *Why it helps:* the same bias-variance dial as $\lambda$-returns, applied directly to the advantage; $\lambda\in[0.9, 0.97]$ is the empirical sweet spot in continuous-control benchmarks.

- **Orthogonal weight initialisation, gain $\sqrt{2}$** *(Saxe et al., 2014; ICLR-blog standard for PPO/A2C)* — orthogonal init on hidden layers, gain $0.01$ on the policy logits head, gain $1.0$ on the value head. *Why it helps:* the tiny gain on the logits keeps the initial policy nearly uniform (high entropy) so exploration starts broad; orthogonal weights preserve gradient norm across depth.

- <a id='trick-4-3-1'></a>**Observation normalisation with running mean/std** — maintain online estimates of per-feature $(\mu, \sigma)$ across all workers and feed $(s - \mu)/\sigma$ to the network. *Why it helps:* removes the dependence on the raw observation scale; critical for MuJoCo-style proprioceptive inputs whose features differ by orders of magnitude.

- <a id='trick-4-3-2'></a>**Advantage normalisation per batch** — $\hat A \leftarrow (\hat A - \mu_A)/(\sigma_A + 10^{-8})$. *Why it helps:* the loss is then scale-invariant in advantage units; this is the *single* most commonly-cited 'magic' trick separating a working A2C/PPO from a broken one.

- <a id='trick-4-3-3'></a>**Global-norm gradient clipping (typically $0.5$)** — clip $\Vert \nabla(\theta,w)\Vert _2$ at a fixed norm before the optimiser step. *Why it helps:* one anomalous rollout (e.g., a sudden mega-reward) cannot blow up the parameters in a single step.

- <a id='trick-4-3-4'></a>**Frame stacking + frame skipping (Atari)** — inherits the four-frame stack and action-repeat-4 pre-processing from DQN. *Why it helps:* provides a velocity signal in pixel input and cuts per-decision compute, exactly as in value-based methods.

<a id='pseudo-4-3'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.3</strong> A2C (synchronous advantage actor–critic) — Raw Pseudocode (without additional tricks) <em>[Mnih et al., 2016 (synchronous variant)]</em>
</div>

Here is the pseudocode for the A2C core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** $\alpha_\theta$, $\alpha_w$, $\gamma$, rollout length $n$, parallel workers $K$, entropy coefficient $\beta$  
Initialize global parameters $\theta$ and $w$  
**repeat** until convergence  
&emsp;*// Each worker collects $n$ steps with current $\pi_\theta$*  
&emsp;**for** $k = 1, 2, \ldots, K$ in parallel **do**  
&emsp;&emsp;**for** $i = 0, 1, \ldots, n - 1$ **do**  
&emsp;&emsp;&emsp;Sample $a_i^{k} \sim \pi(\cdot \mid s_i^{k};  \theta)$; observe $r_{i+1}^{k},  s_{i+1}^{k}$  
&emsp;&emsp;**end for**  
&emsp;**end for**  
&emsp;*// Compute $n$-step returns and advantages*  
&emsp;**for** $k = 1, 2, \ldots, K$ **do**  
&emsp;&emsp;$R \leftarrow 0$ if $s_n^{k}$ terminal else $V(s_n^{k};  w)$  
&emsp;&emsp;**for** $i = n - 1$ **down to** $0$ **do**  
&emsp;&emsp;&emsp;$R \leftarrow r_{i+1}^{k} + \gamma  R$  
&emsp;&emsp;&emsp;$\hat{A}_i^{k} \leftarrow R - V(s_i^{k};  w)$;$\quad R_i^{k} \leftarrow R$  
&emsp;&emsp;**end for**  
&emsp;**end for**  
&emsp;*// Joint update over all $K n$ samples*  
&emsp;$\mathcal{L}_\theta \leftarrow -\dfrac{1}{K n} \sum_{k, i} \bigl[  \log \pi(a_i^{k} \mid s_i^{k};  \theta)  \hat{A}_i^{k} + \beta  \mathcal{H}\bigl(\pi(\cdot \mid s_i^{k};  \theta)\bigr)  \bigr]$  
&emsp;$\mathcal{L}_w \leftarrow \dfrac{1}{K n} \sum_{k, i} \bigl(  R_i^{k} - V(s_i^{k};  w)  \bigr)^{2}$  
&emsp;$\theta \leftarrow \theta - \alpha_\theta  \nabla_{\theta}  \mathcal{L}_\theta;\quad w \leftarrow w - \alpha_w  \nabla_{w}  \mathcal{L}_w$  
**end repeat**  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.3</strong> A2C (synchronous advantage actor–critic) — Pseudocode with Additions (Engineering Tricks) <em>[Mnih et al., 2016 (synchronous variant)]</em>
</div>

Here is the pseudocode for the A2C algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** $\alpha_\theta$, $\alpha_w$, $\gamma$, rollout length $n$, parallel workers $K$, entropy coefficient $\beta$, ✦ GAE $\lambda$ ✦ ([Generalised Advantage Estimation — GAE($\lambda$)](#trick-4-3-1)), ✦ gradient clip norm $c = 0.5$ ✦ ([Global-norm gradient clipping (typically $0.5$)](#trick-4-3-5))  
✦ Initialise shared Actor and Critic with orthogonal weights (gain $\sqrt{2}$ hidden, gain $0.01$ policy head, gain $1.0$ value head) ✦ ([Orthogonal weight initialisation, gain $\sqrt{2}$](#trick-4-3-2))  
✦ Initialise running observation statistics $(\mu_s, \sigma_s)$ across all workers ✦ ([Observation normalisation with running mean/std](#trick-4-3-3))  
**repeat** until convergence  
&emsp;**for** $k = 1, 2, \ldots, K$ in parallel **do**  
&emsp;&emsp;**for** $i = 0, 1, \ldots, n - 1$ **do**  
&emsp;&emsp;&emsp;✦ Pre-process: frame-stack + frame-skip (Atari) or normalise $s \leftarrow (s - \mu_s)/\sigma_s$ ✦ ([Frame stacking + frame skipping (Atari)](#trick-4-3-6) / [Observation normalisation with running mean/std](#trick-4-3-3))  
&emsp;&emsp;&emsp;Sample $a_i^{k} \sim \pi(\cdot \mid s_i^{k};  \theta)$; observe $r_{i+1}^{k},  s_{i+1}^{k}$  
&emsp;&emsp;**end for**  
&emsp;**end for**  
&emsp;✦ *// Compute GAE advantages* ✦ ([Generalised Advantage Estimation — GAE($\lambda$)](#trick-4-3-1))  
&emsp;**for** $k = 1, 2, \ldots, K$ **do**  
&emsp;&emsp;$\delta_i^k \leftarrow r_{i+1}^k + \gamma  V(s_{i+1}^k;  w) - V(s_i^k;  w)$  
&emsp;&emsp;✦ $\hat{A}_i^{k} \leftarrow \sum_{l \geq 0}(\gamma\lambda)^l  \delta_{i+l}^k$ ✦ ([Generalised Advantage Estimation — GAE($\lambda$)](#trick-4-3-1))  
&emsp;&emsp;$R_i^{k} \leftarrow \hat{A}_i^{k} + V(s_i^{k};  w)$  
&emsp;**end for**  
&emsp;✦ $\hat{A} \leftarrow (\hat{A} - \mu_{\hat{A}})/(\sigma_{\hat{A}} + 10^{-8})$ ✦ ([Advantage normalisation per batch](#trick-4-3-4))  
&emsp;$\mathcal{L}_\theta \leftarrow -\dfrac{1}{K n} \sum_{k, i} \bigl[  \log \pi(a_i^{k} \mid s_i^{k};  \theta)  \hat{A}_i^{k} + \beta  \mathcal{H}\bigl(\pi(\cdot \mid s_i^{k};  \theta)\bigr)  \bigr]$  
&emsp;$\mathcal{L}_w \leftarrow \dfrac{1}{K n} \sum_{k, i} \bigl(  R_i^{k} - V(s_i^{k};  w)  \bigr)^{2}$  
&emsp;✦ **if** $\lVert\nabla(\theta,w)\rVert_2 > c$ **then** rescale to norm $c$ ✦ ([Global-norm gradient clipping (typically $0.5$)](#trick-4-3-5))  
&emsp;$\theta \leftarrow \theta - \alpha_\theta  \nabla_{\theta}  \mathcal{L}_\theta;\quad w \leftarrow w - \alpha_w  \nabla_{w}  \mathcal{L}_w$  
**end repeat**  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='a3c'></a>

### A3C (Asynchronous Advantage Actor-Critic)

**Notation:**
- $\theta_\text{local},  w_\text{local}$ — worker-local copies of global $\theta, w$; pulled at the start of each rollout
- $d\theta,  dw$ — gradient increments computed locally and asynchronously pushed to global params
- $K$ — number of parallel asynchronous workers
- $n$ — rollout length (steps each worker collects before pushing $d\theta, dw$)
- $\beta$ — entropy regularisation coefficient in the actor loss
- $H(\pi(\cdot\mid s;\theta_\text{local})) = -\sum_a \pi(a\mid s;\theta_\text{local})\log\pi(a\mid s;\theta_\text{local})$ — policy entropy under the worker's local policy
- $\mathrm{Adv}_i = R_i - V_{w_\text{local}}(s_i)$ — $n$-step advantage at step $i$ within a rollout, with $R_i$ the bootstrapped $n$-step return

<a id='intro-4-4'></a>

**Introduction:** Asynchronous advantage actor–critic. $K$ worker threads run independent environments and ***asynchronously*** push gradient updates to a shared global $(\theta, w)$ — each worker pushes its gradients independently without waiting for others. *Idea:* replace experience replay with parallelism: the diversity of states across asynchronous workers naturally decorrelates updates, and each worker uses an $n$-step advantage as in A2C. *Why it helps (historically):* enabled deep policy-gradient methods on CPU-only multi-core machines pre-GPU-friendly batching; now largely superseded by A2C (simpler, equally fast, no stale gradients).

**Algorithm:**

**Input:** $\alpha_\theta$, $\alpha_w$, $\gamma$, rollout length $n$, parallel workers $K$, entropy coefficient $\beta$

1. Initialize shared (global) Actor $\pi(a \mid s;\theta)$ and Critic $V(s;w)$

2. **For each worker** $k = 1, \ldots, K$ running **asynchronously** in parallel:

   - **Repeat** *(worker loop)*:

     - **Sync:** $\theta_\text{local} \leftarrow \theta$; $\quad w_\text{local} \leftarrow w$ *(pull latest global weights)*

     - **Collect** $n$-step rollout using $\pi(\cdot;\theta_\text{local})$:
       - For $i = 1, \ldots, n$ (or until terminal):
         - $a_i \sim \pi(\cdot \mid s_i;\theta_\text{local})$; take $a_i$, observe $r_i,  s'_i$

     - **Compute $n$-step returns and advantages:**
       $$R \leftarrow \begin{cases} 0 & s_n \text{ is terminal} \\ V(s_n;w_\text{local}) & \text{otherwise (bootstrap)} \end{cases}$$
       - For $i = n{-}1, \ldots, 0$: $\quad R \leftarrow r_{i+1} + \gamma R$; $\quad \mathrm{Adv}_i \leftarrow R - V(s_i;w_\text{local})$

     - **Compute local gradients:**
       $$d\theta \leftarrow \nabla_{\theta_\text{local}}\left[-\sum_i \mathrm{Adv}_i \cdot \log\pi(a_i \mid s_i;\theta_\text{local})  -  \beta  H\bigl(\pi(\cdot \mid s_i;\theta_\text{local})\bigr)\right]$$
       $$dw \leftarrow \nabla_{w_\text{local}}\left[\sum_i \bigl(R - V(s_i;w_\text{local})\bigr)^2\right]$$

     - **Async push to global params** *(no lock — atomic add)*:
       $$\theta \leftarrow \theta - \alpha_\theta  d\theta, \qquad w \leftarrow w - \alpha_w  dw$$

   - Until global step count reaches maximum

3. Return $\pi(\cdot \mid \cdot;\theta)$

✦ ✦ ✦

**Key differences from A2C:**

- **Asynchronous:** each worker updates global params as soon as it finishes its rollout, **without waiting** for other workers. Workers run at different speeds.
- Workers may compute gradients using **stale params** ($\theta_\text{local}$ may be outdated by the time the gradient is applied to $\theta$).
- No replay buffer needed: the diversity of experience from $K$ async workers in different states naturally decorrelates updates.
- In practice, **A2C (synchronous) is often preferred** — simpler, equally fast on GPU, and avoids the stale gradient issue.

<a id='tricks-4-4'></a>

**Additional Known Engineering Tricks**

- **HOGWILD!-style lock-free async updates** *(Recht et al., 2011; adopted by Mnih et al., 2016)* — each worker thread reads $(\theta, w)$ without a lock and atomically adds its gradient. *Why it helps:* eliminates the global synchronisation barrier that bottlenecks A2C, so wall-clock scaling is near-linear in CPU cores. *Interaction:* relies on gradient updates being sparse / commutative-enough that occasional races don't destabilise — the entire raison d'être of A3C over A2C.

- <a id='trick-4-4-1'></a>**Worker-local optimizer state (RMSProp accumulators)** — each thread keeps its own moving-average squared-gradient buffer, but the parameters $\theta$ are global. *Why it helps:* without local optimizer state, every async write contaminates a single global accumulator with stale gradients and the adaptive step size collapses.

- <a id='trick-4-4-2'></a>**Per-worker exploration parameters** — assign each thread a *different* $\varepsilon$ (e.g., sampled from a log-uniform distribution) or different entropy bonus, drawn once per worker lifetime. *Why it helps:* explicit diversity in the data-collecting policies replaces the experience replay that on-policy AC cannot use; the global parameter sees gradients from many exploration regimes simultaneously.

- **LSTM policy for partial observability** *(original A3C paper)* — replace the feed-forward policy head with an LSTM whose hidden state is reset at episode boundaries and carried within a rollout. *Why it helps:* lets the agent remember information beyond the current frame stack; the recurrent state is naturally per-worker. *Interaction:* the asynchronous setup makes recurrent training tractable because gradient noise is high — you cannot afford a tiny effective batch with BPTT on a single worker.

- <a id='trick-4-4-3'></a>**Atomic gradient accumulation** — push gradients via atomic-add CPU instructions rather than locking; on GPU implementations, fall back to small-batch synchronous updates because GPU memory isn't truly shared lock-freely. *Why it helps:* the CPU-friendly version of HOGWILD! that makes A3C a CPU-first algorithm; on GPUs the synchronous A2C variant is almost always preferred.

<a id='pseudo-4-4'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.4</strong> A3C — asynchronous advantage actor–critic (per actor-learner thread) — Raw Pseudocode (without additional tricks) <em>[Mnih et al., 2016]</em>
</div>

Here is the pseudocode for the A3C core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** global shared parameter vectors $\theta$ and $\phi$; global shared counter $T = 0$  
Assume thread-specific parameter vectors $\theta'$ and $\phi'$  
Initialize thread step counter $t \leftarrow 1$  
**repeat**  
&emsp;Reset gradients: $d\theta \leftarrow 0$ and $d\phi \leftarrow 0$  
&emsp;Synchronize thread-specific parameters $\theta' = \theta$ and $\phi' = \phi$  
&emsp;$t_{\text{start}} \leftarrow t$  
&emsp;Get state $s_t$  
&emsp;**repeat**  
&emsp;&emsp;Perform $a_t$ according to policy $\pi(a_t \mid s_t;  \theta')$  
&emsp;&emsp;Receive reward $r_t$ and new state $s_{t+1}$  
&emsp;&emsp;$t \leftarrow t + 1$  
&emsp;&emsp;$T \leftarrow T + 1$  
&emsp;**until** terminal $s_t$ **or** $t - t_{\text{start}} = t_{\max}$  
&emsp;$R \leftarrow 0$ if $s_t$ terminal, else $R \leftarrow V(s_t, \phi')$ *(bootstrap from last state)*  
&emsp;**for** $i \in \lbrace t - 1, \ldots, t_{\text{start}}\rbrace$ **do**  
&emsp;&emsp;$R \leftarrow r_i + \gamma  R$  
&emsp;&emsp;Accumulate gradients wrt $\theta'$: $d\theta \leftarrow d\theta + \nabla_{\theta'} \log \pi(a_i \mid s_i;  \theta') \bigl(R - V(s_i;  \phi')\bigr)$  
&emsp;&emsp;Accumulate gradients wrt $\phi'$: $d\phi \leftarrow d\phi + \partial\bigl(R - V(s_i;  \phi')\bigr)^{2} / \partial \phi'$  
&emsp;**end for**  
&emsp;Perform asynchronous update of $\theta$ using $d\theta$ and of $\phi$ using $d\phi$ &nbsp;<small>HOGWILD!-style lock-free async update</small>  
**until** $T > T_{\max}$  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.4</strong> A3C — asynchronous advantage actor–critic (per actor-learner thread) — Pseudocode with Additions (Engineering Tricks) <em>[Mnih et al., 2016]</em>
</div>

Here is the pseudocode for the A3C algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** global shared parameter vectors $\theta$ and $\phi$; global shared counter $T = 0$  
✦ Each worker $k$ draws its own $\varepsilon_k \sim \text{LogUniform}$ (or distinct entropy bonus) ✦ ([Per-worker exploration parameters](#trick-4-4-3))  
✦ Each worker maintains local RMSProp accumulators $(g^2_\theta, g^2_\phi)$ ✦ ([Worker-local optimizer state (RMSProp accumulators)](#trick-4-4-2))  
Assume thread-specific parameter vectors $\theta'$ and $\phi'$  
Initialise thread step counter $t \leftarrow 1$  
**repeat**  
&emsp;Reset gradients: $d\theta \leftarrow 0$ and $d\phi \leftarrow 0$  
&emsp;Synchronise thread-specific parameters $\theta' = \theta$ and $\phi' = \phi$  
&emsp;$t_{\text{start}} \leftarrow t$  
&emsp;Get state $s_t$  
&emsp;✦ Reset LSTM hidden state $h$ at episode boundaries; carry within rollout ✦ ([LSTM policy for partial observability](#trick-4-4-4))  
&emsp;**repeat**  
&emsp;&emsp;✦ Perform $a_t$ according to policy $\pi(a_t \mid s_t, h;  \theta')$ *(LSTM policy)* ✦ ([LSTM policy for partial observability](#trick-4-4-4))  
&emsp;&emsp;Receive reward $r_t$ and new state $s_{t+1}$  
&emsp;&emsp;$t \leftarrow t + 1$  
&emsp;&emsp;$T \leftarrow T + 1$  
&emsp;**until** terminal $s_t$ **or** $t - t_{\text{start}} = t_{\max}$  
&emsp;$R \leftarrow 0$ if $s_t$ terminal, else $R \leftarrow V(s_t, \phi')$  
&emsp;**for** $i \in \lbrace t - 1, \ldots, t_{\text{start}}\rbrace$ **do**  
&emsp;&emsp;$R \leftarrow r_i + \gamma  R$  
&emsp;&emsp;Accumulate gradients wrt $\theta'$: $d\theta \leftarrow d\theta + \nabla_{\theta'} \log \pi(a_i \mid s_i;  \theta') \bigl(R - V(s_i;  \phi')\bigr)$  
&emsp;&emsp;Accumulate gradients wrt $\phi'$: $d\phi \leftarrow d\phi + \partial\bigl(R - V(s_i;  \phi')\bigr)^{2} / \partial \phi'$  
&emsp;**end for**  
&emsp;✦ Atomically add $d\theta$ to global $\theta$ and $d\phi$ to global $\phi$ *(lock-free HOGWILD! update via atomic CPU instructions)* ✦ ([HOGWILD!-style lock-free async updates](#trick-4-4-1) / [Atomic gradient accumulation](#trick-4-4-5))  
&emsp;✦ Update worker-local RMSProp accumulators ✦ ([Worker-local optimizer state (RMSProp accumulators)](#trick-4-4-2))  
**until** $T > T_{\max}$  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<div style="margin:28px 0 14px 0; padding:4px 0; border-bottom:1.5px solid #888; font-family:'Latin Modern Roman','Times New Roman',serif;">
<span style="font-size:1.15em; font-weight:800; font-style:italic;">✦&ensp;Advanced Policy Optimization</span>
</div>

<a id='ddpg'></a>

### DDPG (Deep Deterministic Policy Gradient)

*Off-policy, actor-critic for* ***continuous action spaces***. *Combines DPG with DQN tricks (target networks + replay).*
*Like DQN but the $\max_a Q$ is replaced by the actor's output $\mu(s)$, since enumerating $a$ is infeasible in $\mathbb{R}^d$.*

**Notation:**
- $\mu(s;\theta^\mu)$ — **deterministic** actor (policy network); outputs a single continuous action vector, not a distribution
- $Q(s,a;\theta^Q)$ — critic (state-action value network)
- $\theta^{\mu-}, \theta^{Q-}$ — target actor and target critic weights (slow-moving copies of $\theta^\mu, \theta^Q$)
- $\tau$ — Polyak soft-update coefficient (e.g., $\tau = 0.005$): $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$
- $\mathcal{N}_t$ — exploration noise added to actions (e.g., Ornstein-Uhlenbeck or Gaussian)
- $\mathcal{D}, B, M$ — replay buffer, minibatch size, capacity (same as DQN+ER)

<a id='intro-4-5'></a>

**Introduction:** Off-policy actor–critic for ***continuous action spaces***. Combines the deterministic policy gradient (DPG) with DQN-style tricks (target networks + replay buffer). *Idea:* learn a ***deterministic*** actor $\mu(s;\theta^\mu)$ alongside a Q-critic $Q(s,a;\theta^Q)$. Since $\max_a Q$ is intractable in $\mathbb{R}^d$, replace it with $Q(s, \mu(s))$ and update the actor by gradient ascent on this — the chain rule pushes $\mu$ toward actions that the critic rates highly. *Why it helps:* brings DQN-style off-policy learning to continuous control, where Q-learning's $\arg\max$ cannot be enumerated.

**Algorithm:**

**Input:** actor rate $\alpha_\mu$, critic rate $\alpha_Q$, discount $\gamma$, soft-update $\tau$, buffer capacity $M$, minibatch size $B$, episodes $n$

1. Initialize Actor $\mu(s;\theta^\mu)$ and Critic $Q(s,a;\theta^Q)$ with random weights

   Initialize target networks $\theta^{\mu-} \leftarrow \theta^\mu$, $\theta^{Q-} \leftarrow \theta^Q$

   Initialize replay buffer $\mathcal{D}$ with capacity $M$

2. **For** episode $= 1, 2, \ldots, n$ *(outer loop — episodes)*:
   - Initialize $s$ and exploration noise process $\mathcal{N}$
   - **Repeat** for each step *(inner loop — time steps)*:
     - $a \leftarrow \mu(s;\theta^\mu) + \mathcal{N}_t$ *(deterministic action + exploration noise)*
     - Take action $a$, observe $r,  s',  \text{done}$
     - Store $(s, a, r, s', \text{done})$ in $\mathcal{D}$
     - Sample $B$ transitions $\lbrace (s_j, a_j, r_j, s'_j, \text{done}_j)\rbrace$ uniformly from $\mathcal{D}$
     - **Compute critic target** *(uses target actor and target critic)*:
       $$y_j = \begin{cases} r_j & \text{if } \text{done}_j \\ r_j + \gamma  Q\bigl(s'_j,  \mu(s'_j;\theta^{\mu-});  \theta^{Q-}\bigr) & \text{otherwise} \end{cases}$$
     - **Update Critic** *(minimize squared Bellman error)*:
       $$\mathcal{L}(\theta^Q) = \frac{1}{B}\sum_j \bigl(y_j - Q(s_j, a_j;\theta^Q)\bigr)^2, \qquad \theta^Q \leftarrow \theta^Q - \alpha_Q \nabla_{\theta^Q} \mathcal{L}(\theta^Q)$$
     - **Update Actor** *(deterministic policy gradient — chain rule through Q)*:
       $$\nabla_{\theta^\mu} J \approx \frac{1}{B}\sum_j \nabla_a Q(s_j, a;\theta^Q)\big|_{a=\mu(s_j)} \cdot \nabla_{\theta^\mu} \mu(s_j;\theta^\mu)$$
       $$\theta^\mu \leftarrow \theta^\mu + \alpha_\mu \nabla_{\theta^\mu} J$$
     - **Soft-update target networks** *(Polyak averaging)*:
       $$\theta^{Q-} \leftarrow \tau\theta^Q + (1-\tau)\theta^{Q-}, \qquad \theta^{\mu-} \leftarrow \tau\theta^\mu + (1-\tau)\theta^{\mu-}$$
     - $s \leftarrow s'$
   - Until $s$ is terminal

3. Return $\mu(\cdot;\theta^\mu)$

✦ ✦ ✦

**Notes:**

- **Continuous actions:** the $\max_{a'}$ in DQN's Bellman target is replaced by $Q(s', \mu(s';\theta^{\mu-}))$ — the actor *approximates* the argmax over continuous $a$.
- **Deterministic policy gradient** (Silver et al., 2014): for a deterministic $\mu$, the policy gradient simplifies to $\nabla_a Q \cdot \nabla_{\theta^\mu} \mu$ (chain rule through the critic).
- **Exploration is external:** since $\mu$ is deterministic, exploration is injected via additive noise $\mathcal{N}_t$ on the action (Ornstein-Uhlenbeck for temporally correlated noise, or simple Gaussian).
- **Soft target updates:** rather than hard copies every $C$ steps (DQN), target networks slowly track the online networks: $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$.
- **Known weaknesses:** brittle to hyperparameters, **critic Q-value overestimation** (similar to DQN max bias). Fixed by **TD3** (twin critics + delayed actor updates + target action smoothing) and **SAC** (entropy regularisation, stochastic policy).

<a id='tricks-4-5'></a>

**Additional Known Engineering Tricks**

- <a id='trick-4-5-1'></a>**Layer normalisation in critic and actor** — apply LayerNorm after each hidden layer; the original DDPG paper used BatchNorm but it interacts badly with replay (running stats drift). *Why it helps:* normalises feature scales, which is critical because state vectors in continuous control span very different magnitudes per dimension.

- **Parameter-space noise for exploration** *(Plappert et al., 2017)* — perturb $\theta^\mu$ itself with Gaussian noise (re-sampled every $K$ steps) instead of adding $\mathcal{N}_t$ to the action. *Why it helps:* yields state-consistent exploration (the same state evokes the same perturbed action within a rollout), which acts as a primitive over time-correlated noise; combine with action-space noise via an adaptive scale schedule.

- <a id='trick-4-5-2'></a>**Tanh action squashing + reward/state scaling** — pass the actor output through $\tanh$ and rescale to $[a_\text{low}, a_\text{high}]$; centre and unit-scale rewards via running stats. *Why it helps:* keeps actions bounded without clipping (clipping kills the gradient); reward scaling means the same learning rate works across tasks with wildly different reward magnitudes.

- **TD3 extensions** *(Fujimoto et al., 2018)* — three drop-in fixes for DDPG's instability: (i) **twin critics** with target $y = r + \gamma \min_{i=1,2} Q_{\theta_i^-}(s', \mu(s';\theta^{\mu-}))$; (ii) **delayed actor & target updates** every 2 critic steps; (iii) **target action smoothing** $\tilde a = \mu(s';\theta^{\mu-}) + \mathrm{clip}(\epsilon, -c, c)$, $\epsilon\sim\mathcal{N}(0,\sigma_t^2)$. *Why each helps:* twin-min combats critic overestimation; delayed actor updates wait for the critic to catch up before changing the policy; target smoothing forces the critic to be smooth in the action input, killing exploitable sharp critic peaks. *Interaction:* these are essentially *the* fix to DDPG and are almost always applied — bare DDPG is rarely used.

- <a id='trick-4-5-3'></a>**Replay warm-up with random actor** — for the first $\sim10\text{k}$ steps, sample uniformly from the action space instead of using $\mu(s)$. *Why it helps:* the freshly-initialised critic has uninformative Q-values; sampling random actions gives diverse coverage to bootstrap learning, similar to DQN's $\varepsilon=1$ warm-up.

<a id='pseudo-4-5'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.5</strong> DDPG (deep deterministic policy gradient) — Raw Pseudocode (without additional tricks) <em>[Lillicrap et al., 2016]</em>
</div>

Here is the pseudocode for the DDPG core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** actor lr $\alpha_\mu$, critic lr $\alpha_Q$, discount $\gamma$, soft-update $\tau$, buffer capacity $M$, minibatch size $B$, episodes $n$  
Initialize critic $Q(s, a;  \theta^{Q})$ and actor $\mu(s;  \theta^{\mu})$ with random weights  
Initialize target networks: $\theta^{Q'} \leftarrow \theta^{Q}, \theta^{\mu'} \leftarrow \theta^{\mu}$  
Initialize replay buffer $\mathcal{D}$ with capacity $M$  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialize exploration noise process $\mathcal{N}$  
&emsp;Initialize $s$  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;$a \leftarrow \mu(s;  \theta^{\mu}) + \mathcal{N}_t$  
&emsp;&emsp;Take action $a$; observe $r,  s',  \text{done}$  
&emsp;&emsp;Store $(s, a, r, s', \text{done})$ in $\mathcal{D}$  
&emsp;&emsp;Sample minibatch $\lbrace (s_j, a_j, r_j, s'_j, \text{done}_j)\rbrace _{j=1}^{B} \sim \mathcal{D}$  
&emsp;&emsp;$y_j \leftarrow r_j + \gamma (1 - \text{done}_j)  Q\bigl(s'_j,  \mu(s'_j;  \theta^{\mu'});  \theta^{Q'}\bigr)$  
&emsp;&emsp;$\mathcal{L}_Q \leftarrow \dfrac{1}{B} \sum_j \bigl(  y_j - Q(s_j, a_j;  \theta^{Q})  \bigr)^{2}$  
&emsp;&emsp;$\theta^{Q} \leftarrow \theta^{Q} - \alpha_Q  \nabla_{\theta^{Q}}  \mathcal{L}_Q$  
&emsp;&emsp;$\nabla_{\theta^{\mu}} J \approx \dfrac{1}{B} \sum_j \nabla_a  Q(s_j, a;  \theta^{Q})\bigr|_{a = \mu(s_j;  \theta^{\mu})}  \nabla_{\theta^{\mu}}  \mu(s_j;  \theta^{\mu})$  
&emsp;&emsp;$\theta^{\mu} \leftarrow \theta^{\mu} + \alpha_\mu  \nabla_{\theta^{\mu}} J$  
&emsp;&emsp;*// Soft target updates (Polyak averaging)*  
&emsp;&emsp;$\theta^{Q'} \leftarrow \tau  \theta^{Q} + (1 - \tau)  \theta^{Q'}$  
&emsp;&emsp;$\theta^{\mu'} \leftarrow \tau  \theta^{\mu} + (1 - \tau)  \theta^{\mu'}$  
&emsp;&emsp;$s \leftarrow s'$  
&emsp;**until** $s$ is terminal  
**end for**  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.5</strong> DDPG (deep deterministic policy gradient) — Pseudocode with Additions (Engineering Tricks) <em>[Lillicrap et al., 2016]</em>
</div>

Here is the pseudocode for the DDPG algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** actor lr $\alpha_\mu$, critic lr $\alpha_Q$, discount $\gamma$, soft-update $\tau$, buffer capacity $M$, minibatch size $B$, episodes $n$, ✦ warm-up steps $W\approx10\text{k}$ ✦ ([Replay warm-up with random actor](#trick-4-5-5)), ✦ TD3 smoothing noise $\sigma_t$, clip $c$ ✦ ([TD3 extensions](#trick-4-5-4))  
✦ Initialise twin critics $Q_{\theta_1^Q}, Q_{\theta_2^Q}$ with LayerNorm after each hidden layer ✦ ([Layer normalisation in critic and actor](#trick-4-5-1) / [TD3 extensions](#trick-4-5-4))  
✦ Initialise actor $\mu(s; \theta^\mu)$ with LayerNorm; output through $\tanh$ rescaled to $[a_\text{low}, a_\text{high}]$ ✦ ([Layer normalisation in critic and actor](#trick-4-5-1) / [Tanh action squashing + reward/state scaling](#trick-4-5-3))  
Initialise targets: $\theta_i^{Q'} \leftarrow \theta_i^Q, \theta^{\mu'} \leftarrow \theta^\mu$  
Initialise replay buffer $\mathcal{D}$ with capacity $M$  
✦ Maintain running reward/state statistics $(\mu_r,\sigma_r),(\mu_s,\sigma_s)$ ✦ ([Tanh action squashing + reward/state scaling](#trick-4-5-3))  
✦ **for** first $W$ steps **do** sample $a \sim \mathrm{Uniform}(a_\text{low}, a_\text{high})$, store in $\mathcal{D}$ **end for** ✦ ([Replay warm-up with random actor](#trick-4-5-5))  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialise $s$  
&emsp;✦ *(Optionally perturb $\theta^\mu$ with Gaussian noise for parameter-space exploration)* ✦ ([Parameter-space noise for exploration](#trick-4-5-2))  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;✦ $a \leftarrow \tanh\bigl(\mu((s-\mu_s)/\sigma_s; \theta^\mu)\bigr) \cdot \text{scale} + \mathcal{N}_t$ *(or use parameter-space noise)* ✦ ([Tanh action squashing + reward/state scaling](#trick-4-5-3) / [Parameter-space noise for exploration](#trick-4-5-2))  
&emsp;&emsp;Take action $a$; observe $r,  s',  \text{done}$  
&emsp;&emsp;✦ $r \leftarrow r / \sigma_r$ *(reward scaling)* ✦ ([Tanh action squashing + reward/state scaling](#trick-4-5-3))  
&emsp;&emsp;Store $(s, a, r, s', \text{done})$ in $\mathcal{D}$  
&emsp;&emsp;Sample minibatch $\lbrace (s_j, a_j, r_j, s'_j, \text{done}_j)\rbrace _{j=1}^{B} \sim \mathcal{D}$  
&emsp;&emsp;✦ $\tilde{a}' \leftarrow \mu(s'_j; \theta^{\mu'}) + \mathrm{clip}(\epsilon,  -c,  c), \epsilon\sim\mathcal{N}(0,\sigma_t^2)$ *(target action smoothing)* ✦ ([TD3 extensions](#trick-4-5-4))  
&emsp;&emsp;✦ $y_j \leftarrow r_j + \gamma (1-\text{done}_j) \min_{i=1,2} Q_{\theta_i^{Q'}}(s'_j, \tilde{a}')$ *(twin-min target)* ✦ ([TD3 extensions](#trick-4-5-4))  
&emsp;&emsp;**for** $i = 1, 2$ **do** $\theta_i^Q \leftarrow \theta_i^Q - \alpha_Q \nabla_{\theta_i^Q} \frac{1}{B}\sum_j(y_j - Q_{\theta_i^Q}(s_j,a_j))^2$ **end for**  
&emsp;&emsp;✦ **if** $t \bmod 2 = 0$ **then** *(delayed actor + target updates)* ✦ ([TD3 extensions](#trick-4-5-4))  
&emsp;&emsp;&emsp;$\nabla_{\theta^{\mu}} J \approx \dfrac{1}{B} \sum_j \nabla_a  Q_{\theta_1^Q}(s_j, a)\bigr|_{a = \mu(s_j;  \theta^{\mu})}  \nabla_{\theta^{\mu}}  \mu(s_j;  \theta^{\mu})$  
&emsp;&emsp;&emsp;$\theta^{\mu} \leftarrow \theta^{\mu} + \alpha_\mu  \nabla_{\theta^{\mu}} J$  
&emsp;&emsp;&emsp;$\theta_i^{Q'} \leftarrow \tau  \theta_i^Q + (1 - \tau)  \theta_i^{Q'};\quad \theta^{\mu'} \leftarrow \tau  \theta^{\mu} + (1 - \tau)  \theta^{\mu'}$  
&emsp;&emsp;✦ **end if** ✦ ([TD3 extensions](#trick-4-5-4))  
&emsp;&emsp;$s \leftarrow s'$  
&emsp;**until** $s$ is terminal  
**end for**  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='sac'></a>

### SAC (Soft Actor-Critic)

*Off-policy, actor-critic for continuous actions. Builds on DDPG with:* ***stochastic policy***, ***twin critics*** *(min combats overestimation, as in TD3), and a* ***maximum-entropy objective*** *that explicitly rewards exploration.*

**Notation:**
- $\pi(a \mid s;\phi)$ — **stochastic** Gaussian actor (typically $\tanh$-squashed): outputs $(\mu_\phi(s), \sigma_\phi(s))$; action $a = \tanh(\mu_\phi(s) + \sigma_\phi(s) \odot \xi),  \xi \sim \mathcal{N}(0, I)$
- $Q_{\theta_1}, Q_{\theta_2}$ — **twin critics** (two independent Q-networks; using the min combats overestimation)
- $\theta_1^-, \theta_2^-$ — target critic weights (Polyak soft updates with coefficient $\tau$)
- $\alpha$ — **entropy temperature**: trades off expected return vs. policy entropy (can be tuned automatically toward a target entropy $\bar{H}$)
- $H(\pi(\cdot \mid s)) = -\mathbb{E}_{a\sim\pi}[\log\pi(a \mid s)]$ — policy entropy at state $s$
- $\mathcal{D}, B, \tau$ — replay buffer, minibatch size, soft-update coefficient (as in DDPG)

<a id='intro-4-6'></a>

**Introduction:** Off-policy actor–critic for continuous actions. Builds on DDPG with three key changes: a ***stochastic*** (Gaussian, $\tanh$-squashed) actor, ***twin critics*** that take the min to combat overestimation (as in TD3), and a ***maximum-entropy objective*** that explicitly rewards exploration.

**Maximum-entropy objective:**

$$J(\pi) = \mathbb{E}\left[\sum_t \gamma^t \bigl(r_t + \alpha  H(\pi(\cdot \mid s_t))\bigr)\right]$$

*The agent maximises reward* ***plus*** *expected entropy — staying as random as possible while still solving the task. Why it helps:* the stochastic policy explores natively (no external noise needed), twin critics tame the overestimation that destabilises DDPG, and the entropy bonus prevents premature collapse to a deterministic local optimum — yielding state-of-the-art sample efficiency on continuous control.

**Algorithm:**

**Input:** actor rate $\alpha_\phi$, critic rate $\alpha_Q$, discount $\gamma$, soft-update $\tau$, temperature $\alpha$, buffer capacity $M$, minibatch size $B$, episodes $n$

1. Initialize twin critics $Q_{\theta_1}, Q_{\theta_2}$ and stochastic actor $\pi(a \mid s;\phi)$

   Initialize target critics $\theta_1^- \leftarrow \theta_1$, $\theta_2^- \leftarrow \theta_2$

   Initialize replay buffer $\mathcal{D}$ with capacity $M$

2. **For** episode $= 1, 2, \ldots, n$ *(outer loop — episodes)*:
   - Initialize $s$
   - **Repeat** for each step *(inner loop — time steps)*:
     - $a \sim \pi(\cdot \mid s;\phi)$ *(sample from stochastic policy — exploration is built in)*
     - Take action $a$, observe $r,  s',  \text{done}$; store $(s, a, r, s', \text{done})$ in $\mathcal{D}$
     - Sample $B$ transitions $\lbrace (s_j, a_j, r_j, s'_j, \text{done}_j)\rbrace$ from $\mathcal{D}$

     - **— Critic targets (entropy-augmented, with twin min) —**

     - Sample $\tilde{a}'_j \sim \pi(\cdot \mid s'_j;\phi)$ for each $j$ *(fresh sample from current actor)*
     - $$y_j = r_j + \gamma (1-\text{done}_j)\Bigl[\min_{i=1,2} Q_{\theta_i^-}(s'_j, \tilde{a}'_j)  -  \alpha  \log\pi(\tilde{a}'_j \mid s'_j;\phi)\Bigr]$$

     - **— Update both critics —**

       $$\mathcal{L}(\theta_i) = \frac{1}{B}\sum_j \bigl(y_j - Q_{\theta_i}(s_j, a_j)\bigr)^2,\quad \theta_i \leftarrow \theta_i - \alpha_Q \nabla_{\theta_i} \mathcal{L}(\theta_i),\quad i=1,2$$

     - **— Update actor (reparameterised — gradient flows through sample) —**

     - Sample $\tilde{a}_j = \tanh(\mu_\phi(s_j) + \sigma_\phi(s_j) \odot \xi_j),  \xi_j \sim \mathcal{N}(0,I)$
       $$\mathcal{L}(\phi) = \frac{1}{B}\sum_j \Bigl[\alpha \log\pi(\tilde{a}_j \mid s_j;\phi)  -  \min_{i=1,2} Q_{\theta_i}(s_j, \tilde{a}_j)\Bigr]$$
       $$\phi \leftarrow \phi - \alpha_\phi \nabla_\phi \mathcal{L}(\phi)$$

     - **(Optional) Auto-tune temperature** $\alpha$ toward target entropy $\bar{H}$:
       $$\mathcal{L}(\alpha) = -\alpha \mathbb{E}_{a\sim\pi}\bigl[\log\pi(a \mid s) + \bar{H}\bigr],\qquad \alpha \leftarrow \alpha - \alpha_\alpha \nabla_\alpha \mathcal{L}(\alpha)$$

     - **Soft-update target critics:** $\theta_i^- \leftarrow \tau\theta_i + (1-\tau)\theta_i^-,\quad i=1,2$
     - $s \leftarrow s'$
   - Until $s$ is terminal

3. Return $\pi(\cdot \mid \cdot;\phi)$

✦ ✦ ✦

**Notes:**

- **Stochastic policy** (unlike DDPG): exploration is intrinsic — no need for external action noise. The $\tanh$ squashes the Gaussian sample into a bounded action range.
- **Twin critics + min** (clipped double-Q, as in TD3): $\min(Q_{\theta_1}, Q_{\theta_2})$ in the target combats systematic Q-value overestimation.
- **Reparameterisation trick** ($a = \tanh(\mu + \sigma \odot \xi), \xi\sim\mathcal{N}$): lets gradients flow through the sampled action into the actor, instead of using the high-variance score-function estimator.
- **Entropy term** $-\alpha \log\pi$ inside the critic target and actor loss rewards stochasticity $\Rightarrow$ exploration is principled, not heuristic.
- **Automatic temperature tuning** adjusts $\alpha$ to keep policy entropy near a target $\bar{H}$ (often $\bar{H} = -|A|$) — removes the most sensitive hyperparameter.
- In practice, **SAC is among the most reliable continuous-control algorithms** — strong sample efficiency, robust to hyperparameters.

<a id='tricks-4-6'></a>

**Additional Known Engineering Tricks**

- <a id='trick-4-6-1'></a>**Automatic temperature tuning** — adjust $\alpha$ by gradient descent on $\mathcal{L}(\alpha) = -\alpha \mathbb{E}[\log\pi(a\mid s) + \bar H]$, with target entropy $\bar H = -|A|$. *Why it helps:* removes SAC's single most sensitive hyperparameter — the entropy temperature now adapts to the task automatically. *Interaction:* tied to the reparameterised $\log\pi$ above; the Jacobian-corrected log-prob is what makes $\bar H = -|A|$ a sensible target. *Not shown in the pseudocode (treated as fixed $\alpha$ there); enable as an extra optimiser on a learnable $\log\alpha$.*

- <a id='trick-4-6-2'></a>**Update-to-data (UTD) ratio $\gg 1$** — perform $G$ gradient updates per environment step (typical $G=1$, but $G=20$ in REDQ; recent work goes up to $G=20$ with extra critics). *Why it helps:* SAC's replay buffer hides huge sample efficiency that a single per-step update doesn't extract; increasing $G$ trades wall-clock for sample efficiency at the cost of overfitting to recent buffer contents. *Interaction:* large $G$ amplifies overestimation, so it pairs with more critics or stronger regularisation.

- <a id='trick-4-6-3'></a>**Layer normalisation in critic** — apply LayerNorm after every critic hidden layer. *Why it helps:* essential when twin critics share the same input but must produce diverse estimates — LayerNorm decorrelates their representations and is part of why the min-of-two-Qs is informative rather than redundant.

- <a id='trick-4-6-4'></a>**Random-policy warm-up to fill the buffer** — first $\sim10\text{k}$ env steps act uniformly random. *Why it helps:* exactly as in DDPG/DQN — the policy and critic start nowhere near useful, so random data gives broader bootstrap coverage than the entropy-maximising-but-still-untrained actor would.

- **Reward scaling** *(Haarnoja et al., 2018)* — multiply rewards by a fixed constant ($\sim$ 5 in MuJoCo tasks) before training. *Why it helps:* shifts the relative weight of reward vs. entropy in the objective; equivalent to picking a different fixed $\alpha$, but easier to tune. *Interaction:* superseded by auto-tuned $\alpha$ when that trick is enabled.

<a id='pseudo-4-6'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.6</strong> SAC (soft actor–critic) — Raw Pseudocode (without additional tricks) <em>[Haarnoja et al., 2018]</em>
</div>

Here is the pseudocode for the SAC core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** actor lr $\alpha_\phi$, critic lr $\alpha_Q$, discount $\gamma$, soft-update $\tau$, temperature $\alpha$, buffer $M$, minibatch $B$, episodes $n$  
Initialize twin critics $Q_{\theta_1}, Q_{\theta_2}$ and actor $\pi_\phi$  
Initialize target critic weights $\bar{\theta}_1 \leftarrow \theta_1, \bar{\theta}_2 \leftarrow \theta_2$  
Initialize replay buffer $\mathcal{D}$  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialize $s$  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;Sample $a \sim \pi_\phi(\cdot \mid s)$; take action $a$; observe $r,  s',  \text{done}$  
&emsp;&emsp;Store $(s, a, r, s', \text{done})$ in $\mathcal{D}$  
&emsp;&emsp;Sample minibatch $\lbrace (s_j, a_j, r_j, s'_j, \text{done}_j)\rbrace _{j=1}^{B} \sim \mathcal{D}$  
&emsp;&emsp;*// Critic update: twin-min target with entropy bonus*  
&emsp;&emsp;Sample $\tilde{a}'_j \sim \pi_\phi(\cdot \mid s'_j)$ for each $j$  
&emsp;&emsp;$y_j \leftarrow r_j + \gamma (1 - \text{done}_j) \bigl[  \min_{i = 1, 2} Q_{\bar{\theta}_i}(s'_j,  \tilde{a}'_j) - \alpha  \log \pi_\phi(\tilde{a}'_j \mid s'_j)  \bigr]$  
&emsp;&emsp;**for** $i = 1, 2$ **do**  
&emsp;&emsp;&emsp;$\mathcal{L}_{Q_i} \leftarrow \dfrac{1}{B} \sum_j \bigl(  y_j - Q_{\theta_i}(s_j, a_j)  \bigr)^{2}$  
&emsp;&emsp;&emsp;$\theta_i \leftarrow \theta_i - \alpha_Q  \nabla_{\theta_i}  \mathcal{L}_{Q_i}$  
&emsp;&emsp;**end for**  
&emsp;&emsp;*// Actor update via reparameterisation: $\tilde{a}_j = f_\phi(\varepsilon_j;  s_j)$*  
&emsp;&emsp;Sample $\tilde{a}_j \sim \pi_\phi(\cdot \mid s_j)$ for each $j$  
&emsp;&emsp;$\mathcal{L}_\pi \leftarrow \dfrac{1}{B} \sum_j \bigl[  \alpha  \log \pi_\phi(\tilde{a}_j \mid s_j) - \min_{i = 1, 2} Q_{\theta_i}(s_j,  \tilde{a}_j)  \bigr]$  
&emsp;&emsp;$\phi \leftarrow \phi - \alpha_\phi  \nabla_{\phi}  \mathcal{L}_\pi$  
&emsp;&emsp;*// Soft target updates*  
&emsp;&emsp;$\bar{\theta}_i \leftarrow \tau  \theta_i + (1 - \tau)  \bar{\theta}_i$ for $i = 1, 2$  
&emsp;&emsp;$s \leftarrow s'$  
&emsp;**until** $s$ is terminal  
**end for**  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.6</strong> SAC (soft actor–critic) — Pseudocode with Additions (Engineering Tricks) <em>[Haarnoja et al., 2018]</em>
</div>

Here is the pseudocode for the SAC algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** actor lr $\alpha_\phi$, critic lr $\alpha_Q$, discount $\gamma$, soft-update $\tau$, ✦ target entropy $\bar{H} = -|A|$ ✦ ([Automatic temperature tuning](#trick-4-6-1)), buffer $M$, minibatch $B$, episodes $n$, ✦ UTD ratio $G$ ✦ ([Update-to-data (UTD) ratio $\gg 1$](#trick-4-6-2)), ✦ warm-up steps $W$ ✦ ([Random-policy warm-up to fill the buffer](#trick-4-6-4))  
✦ Initialise twin critics $Q_{\theta_1}, Q_{\theta_2}$ with LayerNorm after each hidden layer ✦ ([Layer normalisation in critic](#trick-4-6-3))  
Initialise actor $\pi_\phi$  
Initialise target critic weights $\bar{\theta}_1 \leftarrow \theta_1, \bar{\theta}_2 \leftarrow \theta_2$  
✦ Initialise learnable log-temperature $\log\alpha$ ✦ ([Automatic temperature tuning](#trick-4-6-1))  
Initialise replay buffer $\mathcal{D}$  
✦ **for** first $W$ steps **do** sample $a \sim \mathrm{Uniform}$, store in $\mathcal{D}$ **end for** ✦ ([Random-policy warm-up to fill the buffer](#trick-4-6-4))  
**for** $\text{episode} = 1, 2, \ldots, n$ **do**  
&emsp;Initialise $s$  
&emsp;**repeat** (for each step of episode)  
&emsp;&emsp;Sample $a \sim \pi_\phi(\cdot \mid s)$; take action $a$; observe $r,  s',  \text{done}$  
&emsp;&emsp;✦ $r \leftarrow r \cdot \text{rewardScale}$ ✦ ([Reward scaling](#trick-4-6-5))  
&emsp;&emsp;Store $(s, a, r, s', \text{done})$ in $\mathcal{D}$  
&emsp;&emsp;✦ **for** $g = 1, \ldots, G$ **do** *(multiple gradient steps per env step)* ✦ ([Update-to-data (UTD) ratio $\gg 1$](#trick-4-6-2))  
&emsp;&emsp;&emsp;Sample minibatch $\lbrace (s_j, a_j, r_j, s'_j, \text{done}_j)\rbrace _{j=1}^{B} \sim \mathcal{D}$  
&emsp;&emsp;&emsp;$\alpha \leftarrow \exp(\log\alpha)$  
&emsp;&emsp;&emsp;Sample $\tilde{a}'_j \sim \pi_\phi(\cdot \mid s'_j)$ for each $j$  
&emsp;&emsp;&emsp;$y_j \leftarrow r_j + \gamma (1 - \text{done}_j) \bigl[  \min_{i = 1, 2} Q_{\bar{\theta}_i}(s'_j,  \tilde{a}'_j) - \alpha  \log \pi_\phi(\tilde{a}'_j \mid s'_j)  \bigr]$  
&emsp;&emsp;&emsp;**for** $i = 1, 2$ **do** $\theta_i \leftarrow \theta_i - \alpha_Q  \nabla_{\theta_i}  \frac{1}{B}\sum_j(y_j - Q_{\theta_i}(s_j, a_j))^{2}$ **end for**  
&emsp;&emsp;&emsp;Sample $\tilde{a}_j \sim \pi_\phi(\cdot \mid s_j)$ for each $j$  
&emsp;&emsp;&emsp;$\mathcal{L}_\pi \leftarrow \dfrac{1}{B} \sum_j \bigl[  \alpha  \log \pi_\phi(\tilde{a}_j \mid s_j) - \min_{i = 1, 2} Q_{\theta_i}(s_j,  \tilde{a}_j)  \bigr]$  
&emsp;&emsp;&emsp;$\phi \leftarrow \phi - \alpha_\phi  \nabla_{\phi}  \mathcal{L}_\pi$  
&emsp;&emsp;&emsp;✦ $\log\alpha \leftarrow \log\alpha - \alpha_\alpha  \nabla_{\log\alpha}\bigl(-\alpha (\log\pi_\phi(\tilde{a}_j \mid s_j) + \bar{H})\bigr)$ *(auto-tune temperature)* ✦ ([Automatic temperature tuning](#trick-4-6-1))  
&emsp;&emsp;&emsp;$\bar{\theta}_i \leftarrow \tau  \theta_i + (1 - \tau)  \bar{\theta}_i$ for $i = 1, 2$  
&emsp;&emsp;✦ **end for** ✦ ([Update-to-data (UTD) ratio $\gg 1$](#trick-4-6-2))  
&emsp;&emsp;$s \leftarrow s'$  
&emsp;**until** $s$ is terminal  
**end for**  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

<a id='ppo'></a>

### PPO (Proximal Policy Optimization)

*Model-free, on-policy, policy gradient; updates from* ***fixed-length rollouts*** *(no need for full episodes). The de-facto default for deep RL —* ***simple, stable, sample-efficient***.
*Solves the core trust-region problem (TRPO) with a much simpler* ***clipped surrogate objective*** *that prevents destructively large policy updates.*

**Notation:**
- $\pi_\theta, \pi_{\theta_\text{old}}$ — current and previous policies (rollouts are collected with $\pi_{\theta_\text{old}}$, the fixed snapshot at the start of the update)
- $r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$ — **importance ratio** between new and old policies
- $\hat{A}_t$ — advantage `estimate` at step $t$ (typically **GAE** — Generalized Advantage Estimation)
- $\epsilon$ — **clip range** (e.g., $0.2$): how far $r_t(\theta)$ is allowed to drift from $1$
- $K$ — number of optimisation epochs per data batch (each batch reused $K$ times)
- $TD(0)$ (TD error $\delta_t$) -- the one-step estimate, low variance & high bias <br/> $A_t \approx r_t + \gamma V(s_{t+1}) - V(s_t)$
- $MC$ -- Monte Carlo, the full-trajectory estimate, low bias & high variance <br/>$A_t \approx \left( r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots \right) - V(s_t)$

- $\lambda$ — GAE smoothing parameter $\in[0,1]$ interpolates (tradeoff) between $TD(0)$ (bias) and MC (variance) <br/><br/>Same as `A2C` above:
- $\beta$ — entropy regularisation coefficient added to PPO's clipped actor loss to keep $\pi_\theta$ stochastic
- $H(\pi(\cdot\mid s;\theta)) = -\sum_a \pi(a\mid s;\theta)\log\pi(a\mid s;\theta)$ — policy entropy at state $s$; higher $\Rightarrow$ more uniform $\pi_\theta \Rightarrow$ more exploration
- $K_\text{workers}$ — number of parallel environment workers each rolling out under $\pi_{\theta_\text{old}}$ (renamed from A2C's $K$ to disambiguate from PPO's $K$ epochs over the collected rollout)
- $n$ — rollout length per worker before each PPO update phase (so the per-update batch contains $K_\text{workers}\cdot n$ transitions)

<a id='intro-4-7'></a>

**Introduction:** 

Model-free, on-policy policy gradient that updates from ***fixed-length rollouts*** (no need for full episodes). The de-facto default for deep RL — ***simple, stable, sample-efficient***. *Idea:* solve the core trust-region problem of TRPO with a much simpler ***clipped surrogate objective*** that prevents destructively large policy updates:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\bigl(r_t(\theta) \hat{A}_t,   \mathrm{clip}\bigl(r_t(\theta),  1-\epsilon,  1+\epsilon\bigr) \hat{A}_t\bigr)\right]$$

*Caps the incentive to push the importance ratio $r_t(\theta) = \pi_\theta/\pi_{\theta_\text{old}}$ beyond $1\pm\epsilon$:* ***the min with the clipped term removes the gradient*** *as soon as the policy moves too far in either direction. Why it helps:* preserves the “small step” property that makes TRPO reliable while replacing its constrained optimisation with a single scalar clip — robust, fewer hyperparameters, and reusable data via several epochs ($K$) per rollout.

**Algorithm:**

**Input:** $\alpha_\theta$, $\alpha_w$, $\gamma$, GAE $\lambda$, clip $\epsilon$, epochs $K$, minibatch size $B$, workers $K_\text{workers}$, rollout length $n$, entropy coefficient $\beta$

1. Initialize Actor $\pi_\theta$ and Critic $V_w$

2. **Repeat** until convergence *(outer loop — updates)*:

   - **— Collect rollout with the** ***current*** **policy** $\pi_{\theta_\text{old}} \leftarrow \pi_\theta$ —

   - For each worker $k = 1, \ldots, K_\text{workers}$, collect $n$ steps using $\pi_{\theta_\text{old}}$:
     - Store transitions $(s_t, a_t, r_t, s_{t+1},  \log\pi_{\theta_\text{old}}(a_t \mid s_t),  V_w(s_t))$

   - **— Compute advantages via GAE($\lambda$) and returns —**

   - For $t = n{-}1, \ldots, 0$ (backwards):
     $$\delta_t = r_t + \gamma  V_w(s_{t+1}) - V_w(s_t), \qquad \hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$$
     $$\hat{R}_t = \hat{A}_t + V_w(s_t) \quad \text{(target for critic)}$$

   - (Optionally) normalise $\hat{A}_t$ across the batch: $\hat{A}_t \leftarrow (\hat{A}_t - \mu_A)/\sigma_A$

   - **— Optimise for** $K$ **epochs over the** ***same*** **batch —**

   - **For** epoch $= 1, \ldots, K$:
     - Shuffle and split batch into minibatches of size $B$
     - For each minibatch:
       - Compute ratio: $r_t(\theta) = \exp\bigl(\log\pi_\theta(a_t \mid s_t) - \log\pi_{\theta_\text{old}}(a_t \mid s_t)\bigr)$
       - **Clipped policy loss:**
         $$\mathcal{L}^{\text{CLIP}}(\theta) = -\frac{1}{B}\sum_t \min\Bigl(r_t(\theta) \hat{A}_t,   \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\Bigr)$$
       - **Value loss:**
         $$\mathcal{L}^{\text{V}}(w) = \frac{1}{B}\sum_t \bigl(V_w(s_t) - \hat{R}_t\bigr)^2$$
       - **Entropy bonus:** $\mathcal{L}^{\text{H}}(\theta) = -\dfrac{1}{B}\sum_t H\bigl(\pi_\theta(\cdot \mid s_t)\bigr)$
       - **Total loss:** $\mathcal{L}(\theta, w) = \mathcal{L}^{\text{CLIP}}(\theta) + c_v \mathcal{L}^{\text{V}}(w) + \beta \mathcal{L}^{\text{H}}(\theta)$
       - Gradient step: $\theta \leftarrow \theta - \alpha_\theta \nabla_\theta \mathcal{L},   w \leftarrow w - \alpha_w \nabla_w \mathcal{L}$

3. Return $\pi_\theta$

✦ ✦ ✦

**Why the clip works — intuition:**

- If $\hat{A}_t > 0$ (action was better than average): we want to **increase** $\pi_\theta(a_t \mid s_t)$. The clip caps the surrogate at $r_t = 1+\epsilon$ $\Rightarrow$ no incentive to push the ratio further $\Rightarrow$ gradient vanishes once $r_t > 1+\epsilon$.
- If $\hat{A}_t < 0$ (action was worse): we want to **decrease** $\pi_\theta(a_t \mid s_t)$. The clip caps it at $r_t = 1-\epsilon$ $\Rightarrow$ no incentive to push $r_t$ below $1-\epsilon$.
- The $\min$ ensures the clipping only kicks in when it would make the objective *easier* — i.e., it's a **lower bound** on the unclipped surrogate, so optimisation is conservative.

**Notes:**

- **On-policy** but reuses each batch for $K$ epochs (a few — typically $K = 4$-$10$) $\Rightarrow$ much more sample-efficient than vanilla policy gradient / A2C without the instability of off-policy methods.
- **GAE($\lambda$)** smoothly interpolates between TD$(0)$ advantage ($\lambda = 0$, low variance, high bias) and Monte Carlo advantage ($\lambda = 1$, high variance, low bias). Typical $\lambda \in [0.9, 0.97]$.
- **No second-order optimisation** (unlike TRPO): just SGD/Adam on the clipped objective $\Rightarrow$ simple to implement, well-suited to GPUs.
- An older PPO variant uses an **adaptive KL penalty** $-\beta_{KL} \mathrm{KL}(\pi_{\theta_\text{old}} \Vert  \pi_\theta)$ instead of the clip; the clip variant is now standard.
- **Works for both discrete and continuous actions** (Gaussian policy for continuous) — one of the few RL algorithms that generalises across action spaces with minimal tuning.
- In practice, **PPO is the default first choice** for new problems: robust, well-understood, strong baselines (used in OpenAI Five, ChatGPT RLHF, etc.).

<a id='tricks-4-7'></a>

**Additional Known Engineering Tricks**

- **GAE($\lambda$)** *(Schulman et al., 2016)* — replace the n-step advantage shown in the pseudocode with $\hat A_t = \sum_{l\geq 0} (\gamma\lambda)^l \delta_{t+l}$, where $\delta_t = r_t + \gamma V(s'_t;w) - V(s_t;w)$. *Why it helps:* lower-variance advantage than Monte-Carlo returns, lower-bias than one-step TD; typical $\lambda \in [0.9, 0.97]$ is the empirical sweet spot. *Interaction:* the resulting $\hat A$ values are normalised next.

- <a id='trick-4-7-1'></a>**Advantage normalisation per minibatch** — $\hat A \leftarrow (\hat A - \mu_{\hat A})/(\sigma_{\hat A} + 10^{-8})$. *Why it helps:* the loss is scale-invariant in advantage units, so the *same* clip range $\epsilon=0.2$ works across all environments. *Interaction:* must be done *after* advantage computation but *before* the SGD epochs, so the same mean/std is reused within each batch.

- <a id='trick-4-7-2'></a>**Value-loss clipping** — clip $V_w(s_t)$ around the old value: $V^\text{clip} = V_\text{old} + \mathrm{clip}(V_w - V_\text{old}, -\epsilon, +\epsilon)$, then take $\max((V_w - R_t)^2, (V^\text{clip} - R_t)^2)$. *Why it helps:* mirrors the policy clip on the critic so a single bad batch can't drag $V$ wildly; controversial — *(Andrychowicz et al., 2021)* found ablating it harmless on most tasks.

- <a id='trick-4-7-3'></a>**Global-norm gradient clipping, threshold $\sim 0.5$** — clip $\Vert \nabla(\theta,w)\Vert _2$ before each Adam step. *Why it helps:* the multi-epoch updates over a single batch can accumulate large gradients on the last epoch when the ratio $r_t$ moves furthest; clipping caps the per-step damage.

- <a id='trick-4-7-4'></a>**Orthogonal weight initialisation, scaled per-head** — orthogonal init with gain $\sqrt 2$ on hidden layers, **gain 0.01 on the policy logits head**, gain 1.0 on the value head. *Why it helps:* the tiny gain on the policy head ensures the initial policy is nearly uniform $\Rightarrow$ high initial entropy $\Rightarrow$ broad initial exploration. This is one of the well-known *implementation details* that separates a working PPO from a non-working one (see *Engstrom et al., 2020* and *Andrychowicz et al., 2021*).

- <a id='trick-4-7-5'></a>**Linear learning-rate annealing** — $\alpha_t = \alpha_0 \cdot (1 - t/T)$. *Why it helps:* the same recipe as supervised learning — large steps early, fine steps late; reduces the risk of late-training instability when the policy is near-converged.

- <a id='trick-4-7-6'></a>**Observation / reward normalisation with running stats** — maintain running $(\mu, \sigma)$ over all parallel envs, feed $(s-\mu)/\sigma$ to the network, and divide rewards by the running std of discounted returns. *Why it helps:* makes a single hyperparameter set work across tasks with completely different observation and reward scales — a recurring theme; PPO without these is the canonical example of brittle deep RL.

- <a id='trick-4-7-7'></a>**Early stopping by KL divergence** — abort the $K$-epoch loop if $\mathbb{E}_t[\mathrm{KL}(\pi_{\theta_\text{old}} \Vert  \pi_\theta)] > 1.5 \delta_\text{target}$. *Why it helps:* the clip alone is not enough when the advantage sign is mixed; an explicit KL target gives a hard guard against destructive updates. *Interaction:* combines naturally with the clip — the clip handles the typical case and KL early-stop catches the rare runaway minibatch.

<a id='pseudo-4-7'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.7</strong> PPO with clipped surrogate objective — Raw Pseudocode (without additional tricks) <em>[Schulman et al., 2017]</em>
</div>

Here is the pseudocode for the PPO core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** $\alpha_\theta$, $\alpha_w$, $\gamma$, clip $\epsilon$, epochs $K$, minibatch size $B$, workers $K_{\text{w}}$, rollout length $n$, entropy coefficient $\beta$, value coefficient $c_v$  
Initialize policy parameters $\theta$ and value parameters $w$  
**repeat** until convergence  
&emsp;$\theta_{\text{old}} \leftarrow \theta$  
&emsp;**for each** worker $k = 1, \ldots, K_{\text{w}}$ **do**  
&emsp;&emsp;Collect $n$ steps using $\pi_{\theta_{\text{old}}}$: $(s_i, a_i, r_i, s'_i)$ for $i = 0, \ldots, n - 1$  
&emsp;**end for**  
&emsp;*// Bootstrapped n-step return and advantage*  
&emsp;$R \leftarrow 0$ if $s'_{n-1}$ terminal else $V(s'_{n-1};  w)$  
&emsp;**for** $i = n - 1$ **down to** $0$ **do**  
&emsp;&emsp;$R \leftarrow r_i + \gamma  R$  
&emsp;&emsp;$R_i \leftarrow R;\quad \hat{A}_i \leftarrow R_i - V(s_i;  w)$  
&emsp;**end for**  
&emsp;*// Multi-epoch minibatch updates*  
&emsp;**for** $\text{epoch} = 1, 2, \ldots, K$ **do** &nbsp;<small>multi-epoch reuse of one rollout</small>  
&emsp;&emsp;**for each** minibatch of size $B$ **do**  
&emsp;&emsp;&emsp;$r_t(\theta) \leftarrow \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$  
&emsp;&emsp;&emsp;$\mathcal{L}^{\text{CLIP}} \leftarrow \mathbb{E}_t\bigl[  \min\bigl(r_t(\theta)  \hat{A}_t,   \mathrm{clip}(r_t(\theta),  1 - \epsilon,  1 + \epsilon)  \hat{A}_t\bigr)  \bigr]$  
&emsp;&emsp;&emsp;$\mathcal{L}^{V} \leftarrow \mathbb{E}_t\bigl[  (R_t - V(s_t;  w))^{2}  \bigr]$  
&emsp;&emsp;&emsp;$\mathcal{L}^{H} \leftarrow \mathbb{E}_t\bigl[  \mathcal{H}\bigl(\pi_\theta(\cdot \mid s_t)\bigr)  \bigr]$  
&emsp;&emsp;&emsp;$\mathcal{L} \leftarrow - \mathcal{L}^{\text{CLIP}} + c_v  \mathcal{L}^{V} - \beta  \mathcal{L}^{H}$  
&emsp;&emsp;&emsp;$\theta \leftarrow \theta - \alpha_\theta  \nabla_{\theta}  \mathcal{L};\quad w \leftarrow w - \alpha_w  \nabla_{w}  \mathcal{L}$  
&emsp;&emsp;**end for**  
&emsp;**end for**  
**end repeat**  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.7</strong> PPO with clipped surrogate objective — Pseudocode with Additions (Engineering Tricks) <em>[Schulman et al., 2017]</em>
</div>

Here is the pseudocode for the PPO algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** ✦ initial lr $\alpha_0$ ✦ ([Linear learning-rate annealing](#trick-4-7-6)), $\alpha_w$, $\gamma$, clip $\epsilon$, epochs $K$, minibatch size $B$, workers $K_{\text{w}}$, rollout length $n$, entropy coefficient $\beta$, value coefficient $c_v$, ✦ GAE $\lambda$ ✦ ([GAE($\lambda$)](#trick-4-7-1)), ✦ gradient clip $c = 0.5$ ✦ ([Global-norm gradient clipping, threshold $\sim 0.5$](#trick-4-7-4)), ✦ KL target $\delta_\text{KL}$ ✦ ([Early stopping by KL divergence](#trick-4-7-8))  
✦ Initialise Actor and Critic with orthogonal weights (gain $\sqrt{2}$ hidden, gain $0.01$ policy logits head, gain $1.0$ value head) ✦ ([Orthogonal weight initialisation, scaled per-head](#trick-4-7-5))  
✦ Initialise running observation stats $(\mu_s,\sigma_s)$ and reward normaliser ✦ ([Observation / reward normalisation with running stats](#trick-4-7-7))  
**repeat** until convergence  
&emsp;✦ $\alpha_t \leftarrow \alpha_0 \cdot (1 - t/T)$ *(linear lr annealing)* ✦ ([Linear learning-rate annealing](#trick-4-7-6))  
&emsp;$\theta_{\text{old}} \leftarrow \theta$  
&emsp;**for each** worker $k = 1, \ldots, K_{\text{w}}$ **do**  
&emsp;&emsp;✦ Normalise observations: $s \leftarrow (s - \mu_s)/\sigma_s$ ✦ ([Observation / reward normalisation with running stats](#trick-4-7-7))  
&emsp;&emsp;Collect $n$ steps using $\pi_{\theta_{\text{old}}}$: $(s_i, a_i, r_i, s'_i)$ for $i = 0, \ldots, n - 1$  
&emsp;**end for**  
&emsp;✦ *// Compute GAE advantages* ✦ ([GAE($\lambda$)](#trick-4-7-1))  
&emsp;$\delta_i \leftarrow r_i + \gamma  V(s'_i;  w) - V(s_i;  w)$  
&emsp;✦ $\hat{A}_i \leftarrow \sum_{l \geq 0}(\gamma\lambda)^l  \delta_{i+l}$ ✦ ([GAE($\lambda$)](#trick-4-7-1))  
&emsp;$R_i \leftarrow \hat{A}_i + V(s_i;  w)$  
&emsp;✦ $\hat{A} \leftarrow (\hat{A} - \mu_{\hat{A}})/(\sigma_{\hat{A}} + 10^{-8})$ *(advantage normalisation)* ✦ ([Advantage normalisation per minibatch](#trick-4-7-2))  
&emsp;*// Multi-epoch minibatch updates*  
&emsp;**for** $\text{epoch} = 1, 2, \ldots, K$ **do**  
&emsp;&emsp;✦ **if** $\mathbb{E}_t[\mathrm{KL}(\pi_{\theta_\text{old}} \Vert  \pi_\theta)] > 1.5 \delta_\text{KL}$ **then break** *(early stopping)* ✦ ([Early stopping by KL divergence](#trick-4-7-8))  
&emsp;&emsp;**for each** minibatch of size $B$ **do**  
&emsp;&emsp;&emsp;$r_t(\theta) \leftarrow \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$  
&emsp;&emsp;&emsp;$\mathcal{L}^{\text{CLIP}} \leftarrow \mathbb{E}_t\bigl[  \min\bigl(r_t(\theta)  \hat{A}_t,   \mathrm{clip}(r_t(\theta),  1 - \epsilon,  1 + \epsilon)  \hat{A}_t\bigr)  \bigr]$  
&emsp;&emsp;&emsp;✦ $V^{\text{clip}} \leftarrow V_\text{old} + \mathrm{clip}(V_w - V_\text{old},  -\epsilon,  +\epsilon)$ ✦ ([Value-loss clipping](#trick-4-7-3))  
&emsp;&emsp;&emsp;✦ $\mathcal{L}^{V} \leftarrow \mathbb{E}_t\bigl[  \max\bigl((V_w - R_t)^{2},  (V^{\text{clip}} - R_t)^{2}\bigr)  \bigr]$ ✦ ([Value-loss clipping](#trick-4-7-3))  
&emsp;&emsp;&emsp;$\mathcal{L}^{H} \leftarrow \mathbb{E}_t\bigl[  \mathcal{H}\bigl(\pi_\theta(\cdot \mid s_t)\bigr)  \bigr]$  
&emsp;&emsp;&emsp;$\mathcal{L} \leftarrow - \mathcal{L}^{\text{CLIP}} + c_v  \mathcal{L}^{V} - \beta  \mathcal{L}^{H}$  
&emsp;&emsp;&emsp;✦ **if** $\lVert\nabla\mathcal{L}\rVert_2 > c$ **then** rescale to norm $c$ ✦ ([Global-norm gradient clipping, threshold $\sim 0.5$](#trick-4-7-4))  
&emsp;&emsp;&emsp;✦ $\theta \leftarrow \theta - \alpha_t  \nabla_{\theta}  \mathcal{L};\quad w \leftarrow w - \alpha_t  \nabla_{w}  \mathcal{L}$ *(annealed lr)* ✦ ([Linear learning-rate annealing](#trick-4-7-6))  
&emsp;&emsp;**end for**  
&emsp;**end for**  
**end repeat**  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>

✦ ✦ ✦
## Offline RL (Batch RL)
✦ ✦ ✦

<a id='cql'></a>

### CQL (Conservative Q-Learning)

**Notation:**
- $\mathcal{D}$ — fixed offline dataset of transitions (no new environment interactions during training)
- $\lambda$ — CQL regularisation weight; controls conservatism strength
- $\mathcal{L}_\text{CQL}(\theta)$ — conservative penalty: pushes $Q$ down for all actions (via logsumexp), up for actions observed in $\mathcal{D}$
- $\pi_\beta$ — behavior policy: the (unknown) policy that originally collected $\mathcal{D}$

Same as DQN+ER above:
- $\theta^-$ — frozen target network weights (a periodic hard copy of $\theta$; never updated by gradients)
- $\hat{Q}(s,a;\theta^-)$ — target network (separate from the online network $Q$)
- $\mathcal{L}(\theta)$ — loss function (squared Bellman error)
- $B$ — minibatch size (transitions sampled per gradient update)
- $C$ — target update frequency (gradient steps between hard copies $\theta^- \leftarrow \theta$)

<a id='intro-4-8'></a>

**Introduction:** Offline (batch) RL — learns entirely from a ***fixed dataset*** $\mathcal{D}$ with no environment interaction during training. Builds on DQN+ER but adds a ***conservative regulariser*** to combat overestimation of ***out-of-distribution (OOD)*** actions. *Why offline RL is hard:* in online Q-learning, overestimation of $Q$ for unseen actions is self-correcting (the agent tries the action, gets a bad reward, $Q$ is corrected); offline there is no such correction, and the $\max_{a'}$ in the Bellman target latches onto inflated OOD Q-values $\Rightarrow$ divergence. *CQL's fix:* push $Q$ down for *all* actions (via a $\log\sum_a \exp Q$ term) while pushing it back up for actions actually observed in $\mathcal{D}$ — so OOD Q-values become *lower bounds* of their true values and the greedy policy avoids them.

**Algorithm:**

**Input:** fixed dataset $\mathcal{D} = \lbrace (s_j, a_j, r_j, s'_j, \text{done}_j)\rbrace$, discount $\gamma$, learning rate $\alpha$, CQL weight $\lambda$, minibatch size $B$, target update freq $C$, training iterations $n$

1. Initialize **online network** $Q(s,a;\theta)$; **target network** $\hat{Q}(s,a;\theta^-)$ with $\theta^- \leftarrow \theta$

   Initialize step counter $t \leftarrow 0$

2. **For** iteration $= 1, 2, \ldots, n$ *(training loop — no environment interaction)*:

   - Sample minibatch of $B$ transitions $\lbrace (s_j, a_j, r_j, s'_j, \text{done}_j)\rbrace$ from $\mathcal{D}$

   - **Compute standard Bellman targets:**
     $$y_j = \begin{cases} r_j & \text{if } \text{done}_j \\ r_j + \gamma \max_{a'} \hat{Q}(s'_j, a';  \theta^-) & \text{otherwise} \end{cases}$$

   - **Compute CQL conservative penalty:**
     $$\mathcal{L}_\text{CQL}(\theta) = \underbrace{\mathbb{E}_{s \sim \mathcal{D}}\left[\log \sum_a \exp Q(s,a;\theta)\right]}_{\text{push down } Q \text{ for all actions (soft-max)}}  -  \underbrace{\mathbb{E}_{(s,a) \sim \mathcal{D}}\bigl[Q(s,a;\theta)\bigr]}_{\text{push up } Q \text{ for dataset actions}}$$

   - **Compute total loss:**
     $$\mathcal{L}(\theta) = \underbrace{\lambda \mathcal{L}_\text{CQL}(\theta)}_{\text{conservative regulariser}} + \underbrace{\frac{1}{B}\sum_j \bigl(y_j - Q(s_j,a_j;\theta)\bigr)^2}_{\text{standard TD loss}}$$

   - **Gradient update:**
     $$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$$

   - $t \leftarrow t+1$; if $t \bmod C = 0$: $\theta^- \leftarrow \theta$ *(sync target network)*

3. Return $Q(\cdot,\cdot;\theta)$; derive $\pi(s) = \arg\max_a Q(s,a;\theta)$

✦ ✦ ✦

**Why offline RL is hard — and how CQL helps:**

- In standard (online) Q-learning, overestimation of $Q$ for unseen actions is self-correcting: the agent tries the action, gets a bad reward, and $Q$ is corrected. In **offline RL there is no such correction** — the agent never interacts with the environment.
- Naive application of DQN to a fixed dataset leads to severe **overestimation for OOD actions** (actions not well-represented in $\mathcal{D}$). The $\max_{a'}$ in the Bellman target selects these overestimated actions $\Rightarrow$ divergence.
- **CQL's fix:** the conservative penalty $\mathcal{L}_\text{CQL}$ systematically **pushes $Q$-values down** for all state-action pairs (via the logsumexp term) while **pushing them back up** for actions actually observed in the dataset. Net effect: $Q$ values for OOD actions are *lower bounds* of their true values $\Rightarrow$ the policy avoids OOD actions.
- $\lambda$ controls the strength of conservatism: larger $\lambda$ $\Rightarrow$ more conservative (safer but potentially suboptimal); smaller $\lambda$ $\Rightarrow$ closer to standard Q-learning (riskier).

**Notes:**

- **No exploration at all** — the entire algorithm runs on the fixed dataset $\mathcal{D}$.
- $\mathcal{D}$ can come from any source: expert demonstrations, a previous policy, random exploration, or a mix (any behavior policy $\pi_\beta$).
- The logsumexp term $\log\sum_a \exp Q(s,a;\theta)$ is a smooth approximation of $\max_a Q(s,a;\theta)$.
- CQL can also be combined with actor-critic methods (CQL-SAC variant for continuous actions).
- Other offline RL approaches include **BCQ** (constrain policy to dataset support), **TD3+BC** (add behavioral cloning regularisation), and **IQL** (avoid querying OOD actions entirely via expectile regression).

<a id='tricks-4-8'></a>

**Additional Known Engineering Tricks**

- **Lagrangian dual for automatic $\lambda$** *(CQL(H) variant)* — define a constraint $\mathbb{E}_{s\sim\mathcal{D}}[\log\sum_a \exp Q(s,a) - \mathbb{E}_{a\sim\pi_\beta}Q(s,a)] \le \tau$ and update $\lambda$ by dual gradient ascent on the violation. *Why it helps:* removes the single most sensitive hyperparameter — $\lambda$ now auto-adapts to the dataset's coverage; too-conservative when off-support actions get high $Q$, relaxes once they don't.

- <a id='trick-4-8-1'></a>**CQL-SAC: combine with twin critics + entropy-regularised actor** — apply the CQL penalty to *both* critics of a SAC base, and let the actor be the entropy-regularised SAC actor. *Why it helps:* offline RL needs continuous-action support; the SAC base brings reparam + min-of-twin Qs, the CQL term brings out-of-distribution conservatism. *Interaction:* this is the de-facto practical variant for continuous-control offline RL benchmarks (D4RL).

- <a id='trick-4-8-2'></a>**Importance-weighted action sampling for the logsumexp** — approximate $\log\sum_a \exp Q(s,a)$ on continuous actions by sampling actions from three sources (uniform, current policy, dataset policy) and combining with importance weights. *Why it helps:* the logsumexp over a continuous action space is intractable; sampling from a mixture covers the relevant support without bias. *Interaction:* the *uniform* branch is what pushes down OOD actions; the *policy* branch focuses computation on the actions actually selected; the *dataset* branch keeps things grounded in observed data.

- <a id='trick-4-8-3'></a>**Behavioural cloning regulariser** — add $-\alpha_\text{BC} \mathbb{E}_{(s,a)\sim\mathcal{D}}[\log\pi(a\mid s)]$ to the actor loss. *Why it helps:* explicitly keeps the learned policy close to $\pi_\beta$ on the dataset's support, complementing CQL's $Q$-side conservatism; widely used in TD3+BC. *Interaction:* with strong CQL the BC term is redundant; with weak CQL it's a useful safety net.

- <a id='trick-4-8-4'></a>**State (and action) normalisation** — apply running-stats normalisation to states and continuous actions before they enter the network. *Why it helps:* offline RL has no exploration to compensate for badly-scaled inputs, so the network must extract every bit of signal from the fixed $\mathcal{D}$; whitening makes that signal scale-invariant.

- <a id='trick-4-8-5'></a>**Layer normalisation throughout** — between every hidden layer of both $Q$-networks and the actor. *Why it helps:* offline training does many gradient steps per data point and is prone to representational collapse; LayerNorm dramatically stabilises this regime — a common ablation finding in modern offline RL papers (e.g., *Ball et al., 2023*).

- <a id='trick-4-8-6'></a>**Min-of-twin Bellman target** — even before CQL, take $\min(\hat Q_{\theta_1^-}, \hat Q_{\theta_2^-})$ in the standard TD target. *Why it helps:* the same overestimation fix that TD3 brings to online RL is *especially* important offline, because the agent never gets to test its overestimates against the real environment.

<a id='pseudo-4-8'></a>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.8</strong> CQL (conservative Q-learning, offline) — Raw Pseudocode (without additional tricks) <em>[Kumar et al., 2020]</em>
</div>

Here is the pseudocode for the CQL core (base) algorithm without any further additions and engineering tricks (also mentioned above) added:

**Input:** fixed dataset $\mathcal{D} = \lbrace (s_j, a_j, r_j, s'_j, \text{done}_j)\rbrace$, discount $\gamma$, learning rate $\alpha$, CQL weight $\lambda$, minibatch size $B$, target update freq $C$, training iterations $n$  
Initialize online network $Q(s, a;  \theta)$; target network $\hat{Q}(s, a;  \theta^{-})$ with $\theta^{-} \leftarrow \theta$  
**for** $t = 1, 2, \ldots, n$ **do**  
&emsp;Sample minibatch $\lbrace (s_j, a_j, r_j, s'_j, \text{done}_j)\rbrace _{j=1}^{B} \sim \mathcal{D}$  
&emsp;*// Standard TD target (data only — no environment interaction)*  
&emsp;$y_j \leftarrow r_j + \gamma (1 - \text{done}_j)  \max_{a'} \hat{Q}(s'_j, a';  \theta^{-})$  
&emsp;$\mathcal{L}_{\text{TD}} \leftarrow \dfrac{1}{B} \sum_j \bigl(  y_j - Q(s_j, a_j;  \theta)  \bigr)^{2}$  
&emsp;*// Conservative regulariser: push down OOD actions, push up data actions*  
&emsp;$\mathcal{L}_{\text{CQL}} \leftarrow \dfrac{1}{B} \sum_j \Bigl[  \log \sum_{a} \exp Q(s_j, a;  \theta)  -  Q(s_j, a_j;  \theta)  \Bigr]$  
&emsp;$\mathcal{L}(\theta) \leftarrow \mathcal{L}_{\text{TD}} + \lambda  \mathcal{L}_{\text{CQL}}$  
&emsp;$\theta \leftarrow \theta - \alpha  \nabla_{\theta}  \mathcal{L}(\theta)$  
&emsp;**if** $t \bmod C = 0$ **then** $\theta^{-} \leftarrow \theta$  
**end for**  

</div>

<div style="border-top:2.5px solid #000; border-bottom:2.5px solid #000; padding:8px 12px; margin:18px 0; line-height:1.55; font-family:'Latin Modern Roman','Times New Roman',serif;">

<div style="border-bottom:1px solid #000; padding-bottom:5px; margin-bottom:8px;">
<strong>Algorithm 4.8</strong> CQL (conservative Q-learning, offline) — Pseudocode with Additions (Engineering Tricks) <em>[Kumar et al., 2020]</em>
</div>

Here is the pseudocode for the CQL algorithm with also employing all of the additions and engineering tricks mentioned above:

**Input:** fixed dataset $\mathcal{D} = \lbrace (s_j, a_j, r_j, s'_j, \text{done}_j)\rbrace$, discount $\gamma$, lr $\alpha$, ✦ initial $\lambda$, constraint $\tau_\text{CQL}$ ✦ ([Lagrangian dual for automatic $\lambda$](#trick-4-8-1)), minibatch $B$, target update freq $C$, iterations $n$, ✦ BC weight $\alpha_\text{BC}$ ✦ ([Behavioural cloning regulariser](#trick-4-8-4)), ✦ entropy temperature $\alpha_\text{ent}$ ✦ ([CQL-SAC: combine with twin critics + entropy-regularised actor](#trick-4-8-2))  
✦ Initialise twin online critics $Q_{\theta_1}, Q_{\theta_2}$ and twin target critics $\hat{Q}_{\theta_1^-}, \hat{Q}_{\theta_2^-}$ with LayerNorm ✦ ([CQL-SAC: combine with twin critics + entropy-regularised actor](#trick-4-8-2) / [Layer normalisation throughout](#trick-4-8-6))  
✦ Initialise entropy-regularised stochastic actor $\pi_\phi$ ✦ ([CQL-SAC: combine with twin critics + entropy-regularised actor](#trick-4-8-2))  
✦ Initialise learnable $\log\lambda$ ✦ ([Lagrangian dual for automatic $\lambda$](#trick-4-8-1))  
✦ Compute running stats $(\mu_s,\sigma_s)$ from $\mathcal{D}$; normalise all states and actions ✦ ([State (and action) normalisation](#trick-4-8-5))  
**for** $t = 1, 2, \ldots, n$ **do**  
&emsp;Sample minibatch $\lbrace (s_j, a_j, r_j, s'_j, \text{done}_j)\rbrace _{j=1}^{B} \sim \mathcal{D}$  
&emsp;*// TD target (CQL-SAC: twin-min + entropy)*  
&emsp;Sample $\tilde{a}' \sim \pi_\phi(\cdot \mid s'_j)$  
&emsp;✦ $y_j \leftarrow r_j + \gamma (1-\text{done}_j)\bigl[\min_{i=1,2}\hat{Q}_{\theta_i^-}(s'_j, \tilde{a}') - \alpha_\text{ent}\log\pi_\phi(\tilde{a}'\mid s'_j)\bigr]$ *(min-of-twin + SAC entropy)* ✦ ([Min-of-twin Bellman target](#trick-4-8-7) / [CQL-SAC: combine with twin critics + entropy-regularised actor](#trick-4-8-2))  
&emsp;$\mathcal{L}_{\text{TD}}^{(i)} \leftarrow \frac{1}{B}\sum_j(y_j - Q_{\theta_i}(s_j, a_j))^2$  
&emsp;*// Conservative regulariser*  
&emsp;✦ Sample actions from three sources: $a^\text{unif}\sim\mathrm{Uniform}$, $a^\pi\sim\pi_\phi(\cdot\mid s_j)$, $a^\beta$ from dataset ✦ ([Importance-weighted action sampling for the logsumexp](#trick-4-8-3))  
&emsp;✦ $\mathcal{L}_\text{CQL}^{(i)} \leftarrow \frac{1}{B}\sum_j\bigl[\log\sum_{\tilde{a}} \exp Q_{\theta_i}(s_j,\tilde{a}) / w(\tilde{a})  -  Q_{\theta_i}(s_j,a_j)\bigr]$ *(IS-weighted logsumexp)* ✦ ([Importance-weighted action sampling for the logsumexp](#trick-4-8-3))  
&emsp;✦ $\lambda \leftarrow \exp(\log\lambda)$; $\log\lambda \leftarrow \log\lambda + \alpha_\lambda (\mathcal{L}_\text{CQL} - \tau_\text{CQL})$ *(dual update)* ✦ ([Lagrangian dual for automatic $\lambda$](#trick-4-8-1))  
&emsp;**for** $i = 1, 2$ **do** $\theta_i \leftarrow \theta_i - \alpha \nabla_{\theta_i}\bigl(\mathcal{L}_\text{TD}^{(i)} + \lambda \mathcal{L}_\text{CQL}^{(i)}\bigr)$ **end for**  
&emsp;*// Actor update (SAC-style + BC)*  
&emsp;Sample $\tilde{a}_j \sim \pi_\phi(\cdot \mid s_j)$  
&emsp;✦ $\mathcal{L}_\pi \leftarrow \frac{1}{B}\sum_j\bigl[\alpha_\text{ent}\log\pi_\phi(\tilde{a}_j\mid s_j) - \min_i Q_{\theta_i}(s_j,\tilde{a}_j)\bigr] - \alpha_\text{BC} \frac{1}{B}\sum_j\log\pi_\phi(a_j\mid s_j)$ *(+ BC regulariser)* ✦ ([CQL-SAC: combine with twin critics + entropy-regularised actor](#trick-4-8-2) / [Behavioural cloning regulariser](#trick-4-8-4))  
&emsp;$\phi \leftarrow \phi - \alpha_\phi  \nabla_{\phi}  \mathcal{L}_\pi$  
&emsp;**if** $t \bmod C = 0$ **then** $\theta_i^{-} \leftarrow \theta_i$ for $i = 1, 2$  
**end for**  

</div>

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

<div style="page-break-after: always; break-after: page;"></div>
