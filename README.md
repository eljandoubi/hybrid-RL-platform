[//]: # (Image References)

[image1]: domain/platform_domain.png "Platform Domain"

# Multi-Head DRL for Parameterised Action Spaces


## Introduction

In this project, I trained two multi-head agents to solve the **Platform Domain** environment.

![Platform Domain][image1]
 
 See [Platform Domain](https://github.com/cycraig/gym-platform) for the details.
 
## Agents

The first is multi-head DQN or MH-DQN for short. It is inspired from [Multi-Pass Q-Networks for Deep Reinforcement Learning with
Parameterised Action Spaces](https://arxiv.org/pdf/1905.04388.pdf).

The second is multi-head TD3 or MH-TD3 for short. It is kind of fusion between MH-DQN and [TD3](https://arxiv.org/pdf/1802.09477.pdf).


## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.


1. Create (and activate) a new environment with Python 3.9.

	- __Linux__ or __Mac__: 
	```bash 
    conda create --name drlnd 
    source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd 
	activate drlnd
	```
    
    
2. Follow the instructions in [Pytorch](https://pytorch.org/) web page to install pytorch and its dependencies (PIL, numpy,...). For __Windows__ or __Linux__ and __cuda 11.7__

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    ```
3. Install matplotlib
    ```bash
    conda install -c conda-forge matplotlib
    ```
4. Follow the instructions in [Platform Domain](https://github.com/cycraig/gym-platform) to install the environment. 

	
5. Clone the repository, and navigate to your working folder.  Then, install several dependencies.
```bash
git clone https://github.com/eljandoubi/hybrid-RL-platfrom.git
cd hybrid-RL-platfrom
```

## Quick Start
(__Make sure that your `path`is deferent from `saved/MH-DQN` and `saved/MH-TD3`__)

You can train, test and visualization an agent by following instructions below.

For MH-DQN agent with default settings and save it your save folder `path` , run :


```bash
python solve_platform.py --path path
```
If you want to switch to MH-TD3 agent, all you need is :

```bash
python solve_platform.py --agent_name MH_TD3 --path path
```

You can choose which task to execute (train, test, visu or all) by the task argument.

For example, to visualize an agent `agent` saved in `path`:
```bash
python solve_platform.py --agent_name agent --task visu --path path
```

For more options, see the help :  

```bash
python solve_platform.py -h
```


## Verification

To evaluate (`task=test`) and/or watch `task=visu` my optimal agent `agent` (the default `--path` contains my saved models):

```bash
python solve_platform.py --agent_name agent --task task
```

To train the optimal MH_TD3 (mean reward : 0.9975) or MH_DQN (mean reward : 0.9998):
```bash
python solve_platform.py --agent_name agent --task train --path path
```
