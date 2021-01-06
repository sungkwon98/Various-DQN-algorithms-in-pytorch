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

variations=[0,0,0,0,0,0]: A list of what variations to use, starting from double, PER, Dueling Network, Multi-step, Distributional, Noisy Net. 0 if not using, 1 if using.

train_param=[0.1,32,1,5000,10]: A list of training parameter, starting from Target Network update rate, Batch size, when to start training (1 means at batch size, 2 means at 2 * batch size), Replay memory size, Target network update frequency.

epsilon_info=[1,0.05,1000]: A list of greedy search information. Epsilon starting value, final value, decay rate respectively.

soft_update=False: Boolean. True if using soft update.

tau=0.9: A value to be used in soft update.

log_every=10: Period of training log.

retrain_save_dir=None: Directory to save the model.

custom_nn=None: Custom neural network.

use_tensor_board=False: Boolean. Uses tensorboard if True.

sw_dir=None: Directory to save tensorboard data.

## Progress

The current version of the uploaded code is unfinished. However, the author does have a near completed version of it. Contact if needed.

