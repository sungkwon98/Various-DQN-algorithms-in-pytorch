import os
import numpy as np
import math
import random
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import helpers
from helpers import Experience_Replay, Prioritized_Experience_Replay
from ANN import CNN, Simple_NN
import gym


class DQN():
    def __init__(self, env, wrap_env=False, wrappers=None, render=False, use_image=False, use_GPU=False,
                 lr=0.001, gamma=0.99, max_ts=100000, variations=[0, 0, 0, 0, 0, 0], train_param=[0.1, 32, 1, 5000, 10],
                 epsilon_info=[1, 0.05, 1000], soft_update=False, tau=0.9,
                 log_every=10, retrain_save_dir=None, custom_nn=None,
                 use_tensor_board=False, sw_dir=None):
        self.env = env
        self.wrap_env = wrap_env
        self.wrappers = wrappers
        if self.wrap_env:
            self.env = helpers.wrap_env(self.env)
        self.state_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.render = render
        self.use_image = use_image
        self.lr = lr
        self.gamma = gamma
        self.max_ts = max_ts
        self.soft_update = soft_update
        self.tau = tau
        self.log_every = log_every
        self.use_tb = use_tensor_board
        self.sw_dir = sw_dir
        self.retrain_save_dir = retrain_save_dir

        if len(variations) == 6:
            self.dqn_type = variations
        else:
            print("Invalid DQN type. Running vanilla DQN.")
            self.dqn_type = [0, 0, 0, 0, 0, 0]
        self.double = True and print("using Double update") if self.dqn_type[0] == 1 else False
        self.per = True and print("using Prioritized Experience Replay") if self.dqn_type[1] == 1 else False
        self.duel = True and print("using Dueling Network") if self.dqn_type[2] == 1 else False
        self.n_step = True and print("using Multi-step update") if self.dqn_type[3] == 1 else False
        self.distribute = True and print("using Distributional RL") if self.dqn_type[4] == 1 else False
        self.noisy = True and print("using Noisy Network") if self.dqn_type[5] == 1 else False

        if len(train_param) == 5:
            self.train_param = train_param
        else:
            self.train_param = [0.1, 32, 1, 5000, 10]
            print("Invalid train_param. Running with default values.")
        self.target_update_rate = self.train_param[0]
        self.batch_size = self.train_param[1]
        self.start_train_batch = self.train_param[2]
        self.replay_size = self.train_param[3]
        self.target_net_update_freq = self.train_param[4]

        if len(epsilon_info) == 3:
            self.epsilon_info = epsilon_info
        else:
            self.epsilon_info = [1, 0.05, 1000]
            print("Invalid epsilon_info. Running with default values.")
        self.epsilon_start = self.epsilon_info[0]
        self.epsilon_end = self.epsilon_info[1]
        self.epsilon_decay = self.epsilon_info[2]

        self.dtype, self.dtype_long, self.using_GPU = self.GPU(use_GPU)
        self.custom_nn = custom_nn
        if self.custom_nn is None:
            if self.use_image:
                self.q_net = CNN(self.state_shape, self.num_actions)
                self.target_net = deepcopy(self.q_net)
            else:
                self.q_net = Simple_NN(self.state_shape, self.num_actions)
                self.target_net = deepcopy(self.q_net)
        else:
            self.q_net = self.custom_nn
            self.target_net = deepcopy(self.q_net)

        if self.using_GPU:
            self.q_net = self.q_net.cuda()
            self.target_net = self.target_net.cuda()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        if self.per:
            self.experience_replay = Prioritized_Experience_Replay(self.replay_size)
        else:
            self.experience_replay = Experience_Replay(self.replay_size)

    def GPU(self, use_GPU):
        CUDA_available = torch.cuda.is_available()
        if CUDA_available and use_GPU:
            GPU_nums = torch.cuda.device_count()
            if GPU_nums == 1:
                GPU_name = torch.cuda.get_device_name(0)
                print("Using 1 GPU: ", GPU_name, ".")
            else:
                print("Using ", GPU_nums, "GPUs.")
                print("This code in not written to be processed with multiple GPUs. Please watch out.")
            dtype = torch.cuda.FloatTensor
            dtype_long = torch.cuda.LongTensor
            using_GPU = True
        elif not CUDA_available:
            print("NOT using GPU: GPU not available.")
            dtype = torch.FloatTensor
            dtype_long = torch.LongTensor
            using_GPU = False
        else:
            print("NOT using GPU: GPU not requested.")
            dtype = torch.FloatTensor
            dtype_long = torch.LongTensor
            using_GPU = False
        return dtype, dtype_long, using_GPU

    def run_train(self):
        if self.use_tb:
            if self.sw_dir is None:
                writer = SummaryWriter()
            else:
                writer = SummaryWriter(self.sw_dir)

        if self.per:
            beta_start = 0.4
            beta_steps = 1000
            beta_by_step = lambda ts: min(1.0, beta_start + ts * (1.0 - beta_start) / beta_steps)

        all_losses, all_rewards = [], []
        episode_reward = 0
        episode_loss = 0
        state = self.env.reset()
        temporary_transition = []

        for ts in range(1, self.max_ts + 1):
            eps = self.get_eps(ts)
            action = self.act(state, eps)
            next_state, reward, done, _ = self.env.step(int(action.cpu()))
            if self.render:
                self.env.render()

            if self.n_step:
                temporary_transition.append((state, action, reward, next_state, done))
                if len(temporary_transition) == 3:
                    # select what to store in the replay buffer
                    state_m = temporary_transition[0][0]
                    action = temporary_transition[0][1]
                    reward_m = temporary_transition[0][2]
                    next_state_m = temporary_transition[0][3]
                    done_m = temporary_transition[0][4]
                    next_reward = temporary_transition[1][2]
                    next_done = temporary_transition[1][4]
                    r2 = temporary_transition[2][2]
                    s3 = temporary_transition[2][3]
                    d2 = temporary_transition[2][4]
                    self.experience_replay.push_3_steps(state_m, action, reward_m, next_state_m, done_m, next_reward,
                                                        next_done, r2, s3, d2)
                    temporary_transition.pop(0)
            else:
                self.experience_replay.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                if self.use_tb:
                    writer.add_scalar("Reward/DQN_Episodic_reward", episode_reward, ts)
                    writer.add_scalar("Loss/DQN_Episodic_Loss", episode_loss, ts)
                episode_reward = 0
                episode_loss = 0

            if self.experience_replay.__len__() > self.start_train_batch * self.batch_size:
                if self.per:
                    beta = beta_by_step(ts)
                loss = self.compute_loss()
                episode_loss += loss.data
                all_losses.append(loss.data)

                # if self.use_tb:
                # writer.add_scalar("Loss/DQN_Loss", loss.data, ts)

                if ts % self.target_net_update_freq == 0:
                    self.soft_net_update() if self.soft_update else self.net_update()

            if ts % self.log_every == 0:
                out_str = "Timestep {}".format(ts)
                if len(all_rewards) > 0:
                    out_str += ", Reward: {}".format(all_rewards[-1])
                if len(all_losses) > 0:
                    out_str += ", Loss: {}".format(all_losses[-1])
                print(out_str)

        if self.render:
            self.env.close()
        # if self.use_tb:
        # SummaryWriter.close()

        if self.retrain_save_dir is not None:
            torch.save({
                'time-step': ts,
                "policy_network_state_dict": self.q_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'experience_replay': self.experience_replay.get_entire_buffer()
            },
                self.retrain_save_dir + '/model_for_retrain_%d.pth' % ts
            )
            print("Model saved!")

    def run_test(self, checkpoint, test_episode=100):
        if self.use_tb:
            if self.sw_dir is None:
                writer = SummaryWriter()
            else:
                writer = SummaryWriter(self.sw_dir)

        if self.custom_nn is None:
            if self.use_image:
                q_net_test = CNN(self.state_shape, self.num_actions)
            else:
                q_net_test = Simple_NN(self.state_shape, self.num_actions)
        else:
            q_net_test = self.custom_nn
        if self.using_GPU:
            q_net_test = q_net_test.cuda()
        q_net_test.load_state_dict(checkpoint["policy_network_state_dict"])
        q_net_test.eval()

        test_rewards = []
        episode_reward = 0
        state = self.env.reset()
        done = False
        for ep in range(1, test_episode + 1):
            while not done:
                action = self.act(state, test_q_net=q_net_test)
                next_state, reward, done, _ = self.env.step(int(action.cpu()))
                if self.render:
                    self.env.render()
                state = next_state
                episode_reward += reward
            state = self.env.reset()
            done = False
            test_rewards.append(episode_reward)
            if self.use_tb:
                writer.add_scalar("Test/DQN_test_reward", episode_reward, ep)
            episode_reward = 0
        print("Model test complete: Test results --> ", test_rewards)

    def get_eps(self, ts):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1 * ts / self.epsilon_decay)

    def act(self, state, eps=0.00, test_q_net=None):
        if test_q_net is None:
            if random.random() > eps:
                state = torch.tensor(np.float32(state)).type(self.dtype).unsqueeze(0)
                q_values = self.q_net.forward(state)
                return q_values.max(1)[1].data[0]
            return torch.tensor(random.randrange(self.num_actions))
        else:
            state = torch.tensor(np.float32(state)).type(self.dtype).unsqueeze(0)
            q_values = test_q_net.forward(state)
            return q_values.max(1)[1].data[0]

    def compute_loss(self):
        if self.n_step:
            if self.per:
                state, action, reward, next_state, done, next_reward, next_done, r2, s3, d2, indices, weights \
                    = self.experience_replay.sample_3_steps(self.batch_size)
            else:
                state, action, reward, next_state, done, next_reward, next_done, r2, s3, d2\
                    = self.experience_replay.sample_3_steps(self.batch_size)
            state = torch.tensor(np.float32(state)).type(self.dtype)
            action = torch.tensor(action).type(self.dtypelong)
            reward = torch.tensor(reward).type(self.dtype)
            next_state = torch.tensor(np.float32(next_state)).type(self.dtype)
            done = torch.tensor(done).type(self.dtype)
            next_reward = torch.tensor(next_reward).type(self.dtype)
            next_done = torch.tensor(next_done).type(self.dtype)
            r2 = torch.tensor(r2).type(self.dtype)
            s3 = torch.tensor(np.float32(s3)).type(self.dtype)
            d2 = torch.tensor(d2).type(self.dtype)
        else:
            if self.per:
                state, action, reward, next_state, done, indices, weights \
                    = self.experience_replay.sample(self.batch_size)
            else:
                state, action, reward, next_state, done = self.experience_replay.sample(self.batch_size)
            state = torch.tensor(np.float32(state)).type(self.dtype)
            next_state = torch.tensor(np.float32(next_state)).type(self.dtype)
            action = torch.tensor(action).type(self.dtype_long)
            reward = torch.tensor(reward).type(self.dtype)
            done = torch.tensor(done).type(self.dtype)

        if self.per:
            weights = torch.tensor(weights).type(self.dtype)

        # Compute Q_values from policy network
        q_values = self.q_net(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        if self.n_step:
            if self.double:
                # n_step double DQN
                online_next_q_values = self.q_net(s3)
                _, max_indices = torch.max(online_next_q_values, dim=1)
                target_q_values = self.target_net(s3)
                next_q_value = torch.gather(target_q_values, 1, max_indices.unsqueeze(1))
                expected_q_value = reward + (1 - done) * (self.gamma * next_reward +
                                    (1 - next_done) * (math.pow(self.gamma, 2) * r2 +
                                    (1 - d2) * (math.pow(self.gamma, 3) * next_q_value.squeeze())))
            else:
                # n_step DQN
                target_q_values = self.target_net(next_state)
                next_q_value = target_q_values.max(1)[0]
                expected_q_value = reward + (1 - done) * (self.gamma * next_reward +
                                    (1 - next_done) * (math.pow(self.gamma, 2) * r2 +
                                    (1 - d2) * (math.pow(self.gamma, 3) * next_q_value.squeeze())))
        else:
            if self.double:
                # double q-learning
                online_next_q_values = self.q_net(next_state)
                _, max_indicies = torch.max(online_next_q_values, dim=1)
                target_q_values = self.target_net(next_state)
                next_q_value = torch.gather(target_q_values, 1, max_indicies.unsqueeze(1))
                expected_q_value = reward + self.gamma * next_q_value.squeeze() * (1 - done)
            else:
                # vanilla DQN
                target_q_values = self.target_net(next_state)
                next_q_value = target_q_values.max(1)[0]
                expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        if self.per:
            loss = (q_value - expected_q_value.data).pow(2) * weights
            prios = loss + 1e-5
            loss = loss.mean()
        else:
            loss = (q_value - expected_q_value.data).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        if self.per:
            self.experience_replay.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        return loss

    def net_update(self):
        for t_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            if t_param is param:
                continue
            new_param = param.data
            t_param.data.copy_(new_param)

    def soft_net_update(self):
        for t_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            if t_param is param:
                continue
            new_param = self.tau * param.data + (1.0 - self.tau) * t_param.data
            t_param.data.copy_(new_param)
