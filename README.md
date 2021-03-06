# Project 3: Colaboration and Competition

### Introduction

![Trained Agent](results/model.gif)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started

1. Create (and activate) a new environment with Python 3.6 via Anaconda.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name your_env_name python=3.6
	source activate your_env_name
	```
	- __Windows__: 
	```bash
	conda create --name your_env_name python=3.6 
	activate your_env_name
	```

2. Clone the repository, and navigate to the python/ folder. Then, install severeral dependencies (you can create env conda with this packages, or install packages in list env_drlnd.yaml.
	```bash
	conda env create -f environment_DRL.yml
	```

3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

4. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 


### Running the algorithm
In order to execute the code, different files are provided:

1. Inside the "src" folder there are different .py functions:
	*  agentDDPG.py : Implementation of DDPG algorithm
	*  learn.py : Unity ML-Agents Toolkit (from Unity)
	*  network.py : Contains the implementation of the Actor/Critic neural networks
	*  setup.py : Script to install the unity dependencies
	*  utils.py : Script to define the different hyperparameters and parameters required to solve the environment

2. In the root foder: 
	* Tennis.ipynb : It contains the code in charge of training the agents, saving the models, saving the results and allowing us to visualize the performance of the agents using Train_mode=False.

So, to start training the models, run the Tennis.ipynb notebook cells to train an agent to solve our required task of playing tennis.

### Results
The results presented in the form of a graph obtained during the training, together with a .txt file where the score is stored, are located in the results folder. 

Two different approaches have been taken:
1. Obtain an average target score of 0.5 over the last 100 episodes. This result is collected in scores_final_0.5.png

2. Obtain an average target score of 2.0 over the last 100 episodes. This result is collected in scores_final_2.0.png


### Models
The different models for each approach are stored in the models folder.

# Additional information
- [Project Report](./Report.md)
