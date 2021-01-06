# Various-DQN-algorithms-in-pytorch
Self implemented DQN algorithms. It uses Pytorch as it's deep learning tool. 

Main part of the DQN algorithm is defined by one class in a file main.py.

This class is customizable so it may fit to various environments with descrete action space. 

Some very common libraries must be pre-installed to run the algorithm. See requirements.txt.

## Class input Variables definition

env: An environment. It must be "gym" format. Eg, env = gym.make("CartPole-v0")
wrap_env=False: Boolean. True if user wants to wrap the environment. Comming soon.
wrappers=None: Custom wrapper class.
render=False: Boolean. Render during training & testing or not.
use_image=False: Boolean. If True, it uses CNN.
use_GPU=False: Boolean. If True, it uses GPU via cuda if available. 
lr=0.001: Learning rate.
gamma=0.99: The gamma factor.
max_ts=100000: Integer. Maximum time-step for training.
variations=[0,0,0,0,0,0]: 
