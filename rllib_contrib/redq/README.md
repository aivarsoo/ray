# RED Q

[Randomized Ensembled Double Q-Learning: Learning Fast Without a Model](https://arxiv.org/abs/2101.05982)  Randomized Ensembled Double Q-Learning (REDQ) extends Soft-Actor Critic (SAC) to ensemble learning.

## Paper abstract
Using a high Update-To-Data (UTD) ratio, model-based methods have recently achieved much higher sample efficiency than previous model-free methods for continuous-action DRL benchmarks. In this paper, we introduce a simple model-free algorithm, Randomized Ensembled Double Q-Learning (REDQ), and show that its performance is just as good as, if not better than, a state-of-the-art model-based algorithm for the MuJoCo benchmark. Moreover, REDQ can achieve this performance using fewer parameters than the model-based method, and with less wall-clock run time. REDQ has three carefully integrated ingredients which allow it to achieve its high performance:

* a UTD ratio >> 1;
* an ensemble of Q functions;
* in-target minimization across a random subset of Q functions from the ensemble.

Through carefully designed experiments, we provide a detailed analysis of REDQ and related model-free algorithms. To our knowledge, REDQ is the first successful model-free DRL algorithm for continuous-action spaces using a UTD ratio >> 1.


## Installation

```
conda create -n rllib-redq python=3.9
conda activate rllib-redq
pip install -r requirements.txt
pip install -e '.[development]'
```

## Usage

[REDQ Example](examples/test_cartpole.py) ...

## TODO

1. Find the reference for critic udapte clipping
1. Figure out the parameter values for batch size and training frequency
