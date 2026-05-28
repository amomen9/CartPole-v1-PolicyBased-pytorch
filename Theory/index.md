<style>
#algo-toc-sidebar{display:block !important;position:fixed;top:0;left:0;width:280px;max-height:100vh;overflow:auto;padding:12px 10px;background:#f7f7f7;border-right:3px solid #000;z-index:1000;font-family:'Latin Modern Roman','Times New Roman',serif;}
#algo-toc-title{font-weight:900;font-size:18px;margin-bottom:10px;}
#algo-toc-sidebar .ind0{margin-left:0;}
#algo-toc-sidebar .ind1{margin-left:14px;}
#algo-toc-sidebar .ind2{margin-left:28px;}
#algo-toc-sidebar .ind3{margin-left:42px;}
#algo-toc-sidebar .group{margin-top:10px;margin-bottom:4px;font-weight:900;font-size:0.95em;}
#algo-toc-sidebar .subgroup{margin-top:6px;margin-bottom:3px;font-weight:800;font-size:0.90em;color:#1a1a1a;}
#algo-toc-sidebar .subsubgroup{margin-top:4px;margin-bottom:2px;font-weight:700;font-style:italic;font-size:0.86em;color:#444;}
#algo-toc-sidebar .leaf{font-weight:700;font-size:0.86em;margin-top:2px;margin-bottom:2px;}
#algo-toc-sidebar .leaf::before{content:'· ';color:#555;font-size:0.8em;}
#algo-toc-sidebar a{color:#000;text-decoration:none;}
#algo-toc-sidebar a:hover{text-decoration:underline;color:#7b1d1d;}
#algo-toc-sidebar details{margin-top:2px;margin-bottom:2px;}
#algo-toc-sidebar summary{font-weight:700;cursor:pointer;list-style:none;padding:1px 0;font-size:0.85em;}
#algo-toc-sidebar summary::-webkit-details-marker{display:none;}
#algo-toc-sidebar summary::before{content:'▸\00a0';color:#555;font-size:0.8em;}
#algo-toc-sidebar details[open] > summary::before{content:'▾\00a0';}
#algo-toc-sidebar details a{display:block;font-weight:400;padding-left:18px;font-size:0.8em;margin:1px 0;color:#333;line-height:1.3;}
#algo-toc-sidebar details a:hover{color:#7b1d1d;}
#notebook-container{padding-left:295px;}
@media (max-width: 950px){#algo-toc-sidebar{position:relative;width:auto;max-height:none;border-right:0;}#notebook-container{padding-left:0;}}
</style>
<div id='algo-toc-sidebar' style='display:none'>
  <div id='algo-toc-title'>Index</div>
  <div class='group ind0'>Model-Based Methods</div>
  <details class='ind3'>
    <summary>1. Value Iteration</summary>
    <a href='#intro-1-1'>· Introduction</a>
    <a href='#value-iteration'>· Algorithm</a>
    <a href='#tricks-1-1'>· Engineering Tricks</a>
    <a href='#pseudo-1-1'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>2. Policy Iteration</summary>
    <a href='#intro-1-2'>· Introduction</a>
    <a href='#policy-iteration'>· Algorithm</a>
    <a href='#tricks-1-2'>· Engineering Tricks</a>
    <a href='#pseudo-1-2'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>3. Held-Karp (Bottom-Up / Tabulation)</summary>
    <a href='#intro-1-3'>· Introduction</a>
    <a href='#held-karp-bottom-up'>· Algorithm</a>
    <a href='#tricks-1-3'>· Engineering Tricks</a>
    <a href='#pseudo-1-3'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>4. Held-Karp (Top-Down / Memoisation)</summary>
    <a href='#intro-1-4'>· Introduction</a>
    <a href='#held-karp-top-down'>· Algorithm</a>
    <a href='#tricks-1-4'>· Engineering Tricks</a>
    <a href='#pseudo-1-4'>· Pseudocode</a>
  </details>
  <div class='group ind0'>Model-Free Methods</div>
  <div class='leaf ind1'><a href='#on-off-policy-intro'>On-Policy vs Off-Policy Algorithms</a></div>
  <div class='leaf ind1'><a href='#full-episodes-intro'>Full Episodes vs Step-wise Updates</a></div>
  <details class='ind3'>
    <summary>1. Monte Carlo (First-Visit, On-Policy)</summary>
    <a href='#intro-2-1'>· Introduction</a>
    <a href='#monte-carlo-first-visit'>· Algorithm</a>
    <a href='#tricks-2-1'>· Engineering Tricks</a>
    <a href='#pseudo-2-1'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>2. Temporal Difference — TD(0)</summary>
    <a href='#intro-2-2'>· Introduction</a>
    <a href='#td-0'>· Algorithm</a>
    <a href='#tricks-2-2'>· Engineering Tricks</a>
    <a href='#pseudo-2-2'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>3. n-Step TD</summary>
    <a href='#intro-2-3'>· Introduction</a>
    <a href='#n-step-td'>· Algorithm</a>
    <a href='#tricks-2-3'>· Engineering Tricks</a>
    <a href='#pseudo-2-3'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>4. SARSA (State-Action-Reward-State-Action)</summary>
    <a href='#intro-2-4'>· Introduction</a>
    <a href='#sarsa'>· Algorithm</a>
    <a href='#tricks-2-4'>· Engineering Tricks</a>
    <a href='#pseudo-2-4'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>5. Q-Learning</summary>
    <a href='#intro-2-5'>· Introduction</a>
    <a href='#q-learning'>· Algorithm</a>
    <a href='#tricks-2-5'>· Engineering Tricks</a>
    <a href='#pseudo-2-5'>· Pseudocode</a>
  </details>
  <div class='subgroup ind1'>Deep RL — Value-Based</div>
  <details class='ind3'>
    <summary>1. DQN Naive (Deep Q-Network, no tricks)</summary>
    <a href='#intro-3-1'>· Introduction</a>
    <a href='#dqn-naive'>· Algorithm</a>
    <a href='#tricks-3-1'>· Engineering Tricks</a>
    <a href='#pseudo-3-1'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>2. DQN + Target Network (DQN+TN)</summary>
    <a href='#intro-3-2'>· Introduction</a>
    <a href='#dqn-target-network'>· Algorithm</a>
    <a href='#tricks-3-2'>· Engineering Tricks</a>
    <a href='#pseudo-3-2'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>3. DQN + Experience Replay (DQN+ER)</summary>
    <a href='#intro-3-3'>· Introduction</a>
    <a href='#dqn-experience-replay'>· Algorithm</a>
    <a href='#tricks-3-3'>· Engineering Tricks</a>
    <a href='#pseudo-3-3'>· Pseudocode</a>
  </details>
  <div class='subgroup ind1'>Deep RL — Policy Gradient Methods</div>
  <div class='leaf ind2'><a href='#vanilla-pg-loss'>What Is the Vanilla Policy Gradient Loss?</a></div>
  <div class='subsubgroup ind2'>Vanilla Policy Gradient</div>
  <details class='ind3'>
    <summary>1. REINFORCE (Monte Carlo Policy Gradient)</summary>
    <a href='#intro-4-1'>· Introduction</a>
    <a href='#reinforce'>· Algorithm</a>
    <a href='#tricks-4-1'>· Engineering Tricks</a>
    <a href='#pseudo-4-1'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>2. Actor-Critic (AC)</summary>
    <a href='#intro-4-2'>· Introduction</a>
    <a href='#actor-critic'>· Algorithm</a>
    <a href='#tricks-4-2'>· Engineering Tricks</a>
    <a href='#pseudo-4-2'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>3. A2C (Advantage Actor-Critic)</summary>
    <a href='#intro-4-3'>· Introduction</a>
    <a href='#a2c'>· Algorithm</a>
    <a href='#tricks-4-3'>· Engineering Tricks</a>
    <a href='#pseudo-4-3'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>4. A3C (Asynchronous Advantage Actor-Critic)</summary>
    <a href='#intro-4-4'>· Introduction</a>
    <a href='#a3c'>· Algorithm</a>
    <a href='#tricks-4-4'>· Engineering Tricks</a>
    <a href='#pseudo-4-4'>· Pseudocode</a>
  </details>
  <div class='subsubgroup ind2'>Advanced Policy Optimization</div>
  <details class='ind3'>
    <summary>1. DDPG (Deep Deterministic Policy Gradient)</summary>
    <a href='#intro-4-5'>· Introduction</a>
    <a href='#ddpg'>· Algorithm</a>
    <a href='#tricks-4-5'>· Engineering Tricks</a>
    <a href='#pseudo-4-5'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>2. SAC (Soft Actor-Critic)</summary>
    <a href='#intro-4-6'>· Introduction</a>
    <a href='#sac'>· Algorithm</a>
    <a href='#tricks-4-6'>· Engineering Tricks</a>
    <a href='#pseudo-4-6'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>3. PPO (Proximal Policy Optimization)</summary>
    <a href='#intro-4-7'>· Introduction</a>
    <a href='#ppo'>· Algorithm</a>
    <a href='#tricks-4-7'>· Engineering Tricks</a>
    <a href='#pseudo-4-7'>· Pseudocode</a>
  </details>
  <details class='ind3'>
    <summary>4. CQL (Conservative Q-Learning)</summary>
    <a href='#intro-4-8'>· Introduction</a>
    <a href='#cql'>· Algorithm</a>
    <a href='#tricks-4-8'>· Engineering Tricks</a>
    <a href='#pseudo-4-8'>· Pseudocode</a>
  </details>
</div>

## Index

**Model-Based Methods**

> > > <details><summary>1. Value Iteration</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-1-1) &nbsp;· [Algorithm](#value-iteration) &nbsp;· [Engineering Tricks](#tricks-1-1) &nbsp;· [Pseudocode](#pseudo-1-1)
> > >
> > > </details>
> > >
> > > <details><summary>2. Policy Iteration</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-1-2) &nbsp;· [Algorithm](#policy-iteration) &nbsp;· [Engineering Tricks](#tricks-1-2) &nbsp;· [Pseudocode](#pseudo-1-2)
> > >
> > > </details>
> > >
> > > <details><summary>3. Held-Karp (Bottom-Up / Tabulation)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-1-3) &nbsp;· [Algorithm](#held-karp-bottom-up) &nbsp;· [Engineering Tricks](#tricks-1-3) &nbsp;· [Pseudocode](#pseudo-1-3)
> > >
> > > </details>
> > >
> > > <details><summary>4. Held-Karp (Top-Down / Memoisation)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-1-4) &nbsp;· [Algorithm](#held-karp-top-down) &nbsp;· [Engineering Tricks](#tricks-1-4) &nbsp;· [Pseudocode](#pseudo-1-4)
> > >
> > > </details>

**Model-Free Methods**

> [On-Policy vs Off-Policy Algorithms](#on-off-policy-intro)
>
> [Full Episodes vs Step-wise Updates](#full-episodes-intro)

> > > <details><summary>1. Monte Carlo (First-Visit, On-Policy)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-2-1) &nbsp;· [Algorithm](#monte-carlo-first-visit) &nbsp;· [Engineering Tricks](#tricks-2-1) &nbsp;· [Pseudocode](#pseudo-2-1)
> > >
> > > </details>
> > >
> > > <details><summary>2. Temporal Difference — TD(0)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-2-2) &nbsp;· [Algorithm](#td-0) &nbsp;· [Engineering Tricks](#tricks-2-2) &nbsp;· [Pseudocode](#pseudo-2-2)
> > >
> > > </details>
> > >
> > > <details><summary>3. n-Step TD</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-2-3) &nbsp;· [Algorithm](#n-step-td) &nbsp;· [Engineering Tricks](#tricks-2-3) &nbsp;· [Pseudocode](#pseudo-2-3)
> > >
> > > </details>
> > >
> > > <details><summary>4. SARSA (State-Action-Reward-State-Action)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-2-4) &nbsp;· [Algorithm](#sarsa) &nbsp;· [Engineering Tricks](#tricks-2-4) &nbsp;· [Pseudocode](#pseudo-2-4)
> > >
> > > </details>
> > >
> > > <details><summary>5. Q-Learning</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-2-5) &nbsp;· [Algorithm](#q-learning) &nbsp;· [Engineering Tricks](#tricks-2-5) &nbsp;· [Pseudocode](#pseudo-2-5)
> > >
> > > </details>

> <strong>Deep RL — Value-Based</strong>

> > > <details><summary>1. DQN Naive (Deep Q-Network, no tricks)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-3-1) &nbsp;· [Algorithm](#dqn-naive) &nbsp;· [Engineering Tricks](#tricks-3-1) &nbsp;· [Pseudocode](#pseudo-3-1)
> > >
> > > </details>
> > >
> > > <details><summary>2. DQN + Target Network (DQN+TN)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-3-2) &nbsp;· [Algorithm](#dqn-target-network) &nbsp;· [Engineering Tricks](#tricks-3-2) &nbsp;· [Pseudocode](#pseudo-3-2)
> > >
> > > </details>
> > >
> > > <details><summary>3. DQN + Experience Replay (DQN+ER)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-3-3) &nbsp;· [Algorithm](#dqn-experience-replay) &nbsp;· [Engineering Tricks](#tricks-3-3) &nbsp;· [Pseudocode](#pseudo-3-3)
> > >
> > > </details>

> <strong>Deep RL — Policy Gradient Methods</strong>

> > [What Is the Vanilla Policy Gradient Loss?](#vanilla-pg-loss)
> >
> > <strong><em>Vanilla Policy Gradient</em></strong>

> > > <details><summary>1. REINFORCE (Monte Carlo Policy Gradient)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-4-1) &nbsp;· [Algorithm](#reinforce) &nbsp;· [Engineering Tricks](#tricks-4-1) &nbsp;· [Pseudocode](#pseudo-4-1)
> > >
> > > </details>
> > >
> > > <details><summary>2. Actor-Critic (AC)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-4-2) &nbsp;· [Algorithm](#actor-critic) &nbsp;· [Engineering Tricks](#tricks-4-2) &nbsp;· [Pseudocode](#pseudo-4-2)
> > >
> > > </details>
> > >
> > > <details><summary>3. A2C (Advantage Actor-Critic)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-4-3) &nbsp;· [Algorithm](#a2c) &nbsp;· [Engineering Tricks](#tricks-4-3) &nbsp;· [Pseudocode](#pseudo-4-3)
> > >
> > > </details>
> > >
> > > <details><summary>4. A3C (Asynchronous Advantage Actor-Critic)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-4-4) &nbsp;· [Algorithm](#a3c) &nbsp;· [Engineering Tricks](#tricks-4-4) &nbsp;· [Pseudocode](#pseudo-4-4)
> > >
> > > </details>

> > <strong><em>Advanced Policy Optimization</em></strong>

> > > <details><summary>1. DDPG (Deep Deterministic Policy Gradient)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-4-5) &nbsp;· [Algorithm](#ddpg) &nbsp;· [Engineering Tricks](#tricks-4-5) &nbsp;· [Pseudocode](#pseudo-4-5)
> > >
> > > </details>
> > >
> > > <details><summary>2. SAC (Soft Actor-Critic)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-4-6) &nbsp;· [Algorithm](#sac) &nbsp;· [Engineering Tricks](#tricks-4-6) &nbsp;· [Pseudocode](#pseudo-4-6)
> > >
> > > </details>
> > >
> > > <details><summary>3. PPO (Proximal Policy Optimization)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-4-7) &nbsp;· [Algorithm](#ppo) &nbsp;· [Engineering Tricks](#tricks-4-7) &nbsp;· [Pseudocode](#pseudo-4-7)
> > >
> > > </details>
> > >
> > > <details><summary>4. CQL (Conservative Q-Learning)</summary>
> > >
> > > &nbsp;&nbsp;&middot; [Introduction](#intro-4-8) &nbsp;· [Algorithm](#cql) &nbsp;· [Engineering Tricks](#tricks-4-8) &nbsp;· [Pseudocode](#pseudo-4-8)
> > >
> > > </details>

<div style="page-break-after: always; break-after: page;"></div>