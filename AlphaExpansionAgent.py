import gym
import gym_alphaexpansion
import math
import random
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from tensorboardX import SummaryWriter


env = gym.make('AlphaExpansion-v0').unwrapped
env.seed(int(round(time.time() * 1000)))
env.reset()
env.render()

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w, input_depth, actions):
        super(DQN, self).__init__()
        conv1_kernel_size = 3
        conv1_stride = 1
        conv1_padding = 1
        conv1_output_depth = 128
        self.conv1 = nn.Conv2d(input_depth, conv1_output_depth,
                               kernel_size=conv1_kernel_size, stride=conv1_stride, padding=conv1_padding)
        self.bn1 = nn.BatchNorm2d(conv1_output_depth)
        conv2_kernel_size = 3
        conv2_stride = 1
        conv2_padding = 1
        conv2_output_depth = 2 * 34  # left right click * all building types
        self.conv2 = nn.Conv2d(conv1_output_depth, conv2_output_depth,
                               kernel_size=conv2_kernel_size, stride=conv2_stride, padding=conv2_padding)
        self.bn2 = nn.BatchNorm2d(conv2_output_depth)
        # conv3_kernel_size = 5
        # conv3_stride = 1
        # conv3_padding = 2
        # conv3_output_depth = 1
        # self.conv3 = nn.Conv2d(conv2_output_depth, conv3_output_depth,
        #                        kernel_size=conv3_kernel_size, stride=conv3_stride, padding=conv3_padding)
        # self.bn3 = nn.BatchNorm2d(conv3_output_depth)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=1, padding=2):
            return ((size - kernel_size + (2 * padding)) // stride) + 1

        # convw = conv2d_size_out(w, kernel_size=conv1_kernel_size, stride=conv1_stride, padding=conv1_padding)
        # convh = conv2d_size_out(h, kernel_size=conv1_kernel_size, stride=conv1_stride, padding=conv1_padding)

        convw = conv2d_size_out(
            conv2d_size_out(w,
                            kernel_size=conv1_kernel_size, stride=conv1_stride, padding=conv1_padding),
            kernel_size=conv2_kernel_size, stride=conv2_stride, padding=conv2_padding)
        convh = conv2d_size_out(
            conv2d_size_out(h,
                            kernel_size=conv1_kernel_size, stride=conv1_stride, padding=conv1_padding),
            kernel_size=conv2_kernel_size, stride=conv2_stride, padding=conv2_padding)

        # convw = conv2d_size_out(
        #     conv2d_size_out(
        #         conv2d_size_out(w,
        #                         kernel_size=conv1_kernel_size, stride=conv1_stride, padding=conv1_padding),
        #         kernel_size=conv2_kernel_size, stride=conv2_stride, padding=conv2_padding),
        #     kernel_size=conv3_kernel_size, stride=conv3_stride, padding=conv3_padding)
        # convh = conv2d_size_out(
        #     conv2d_size_out(
        #         conv2d_size_out(h,
        #                         kernel_size=conv1_kernel_size, stride=conv1_stride, padding=conv1_padding),
        #         kernel_size=conv2_kernel_size, stride=conv2_stride, padding=conv2_padding),
        #     kernel_size=conv3_kernel_size, stride=conv3_stride, padding=conv3_padding)

        linear_input_size = convw * convh * conv2_output_depth
        # self.head = nn.Linear(linear_input_size, actions

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.selu(self.bn1(self.conv1(x)))
        x = F.selu(self.bn2(self.conv2(x)))
        # x = F.selu(self.bn3(self.conv3(x)))
        # return self.head(x.view(x.size(0), -1))
        # return x.view(x.size(0), -1)
        return x






######################################################################
# Training
# --------
#
# Hyperparameters and utilitiesIf you ca
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the durations of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
#


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100000
TARGET_UPDATE = 10
INPUT_HEIGHT = env.game.map.CHUNK_HEIGHT
INPUT_WIDTH = env.game.map.CHUNK_WIDTH
INPUT_DEPTH = 92
ACTIONS = 2 * 34 * INPUT_WIDTH * INPUT_HEIGHT  # 34x28x16x2=30464 possible actions at default
ACTIONS_SHAPE = (2, 34, INPUT_WIDTH, INPUT_HEIGHT)
env.ravel = False
writer = SummaryWriter()

policy_net = DQN(INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH, ACTIONS).to(device)
target_net = DQN(INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH, ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(100000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action_values = policy_net(state)
            one_dimension_action = action_values.view(1, -1).max(1)[1]
            return one_dimension_action.view(1, 1),\
                   np.unravel_index(one_dimension_action.squeeze().cpu(),
                                    ACTIONS_SHAPE)
    else:
        one_dimension_action = torch.tensor([[random.randrange(ACTIONS)]], device=device, dtype=torch.long)
        return one_dimension_action, \
               np.unravel_index(one_dimension_action.squeeze().cpu(),
                                ACTIONS_SHAPE)


def reshape_to_environment(shape):
    new_shape = (2, shape[1]//2, shape[2], shape[3])
    return new_shape


episode_scores = []
episode_rand_prob = []


def plot_scores():
    plt.figure(1)
    plt.clf()
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores_t.numpy())
    # Take 100 episode averages and plot them too
    if len(scores_t) >= 100:
        means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_rand_prob():
    plt.figure(2)
    plt.clf()
    rand_prob_t = torch.tensor(episode_rand_prob, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Probability of Random Action')
    plt.plot(rand_prob_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rand_prob_t) >= 100:
        means = rand_prob_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).view(BATCH_SIZE, -1).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).view(1, -1).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()


def state_preprocess(state):
    return torch.from_numpy(np.dstack((
            state["relative_income"],
            state["terrain"],
            state["buildings"],
            state["building_levels"],
            state["can_upgrade"],
            state["can_build"]
        ))).type(torch.cuda.FloatTensor).permute(2, 0, 1).unsqueeze(0)

######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` Tensor. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes, such as 300+ for meaningful
# duration improvements.
#


num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = state_preprocess(env.reset())
    score = 0.0
    for t in count():
        # Select and perform an action
        one_dimensional_action, high_dimensional_action = select_action(state)
        next_state, reward, done, _ = env.step(high_dimensional_action)
        next_state = state_preprocess(next_state)
        if reward > 0:
            env.render()
        score += reward
        reward = torch.tensor([reward], device=device)

        if done:
            next_state = None
            env.render()

        # Store the transition in memory
        memory.push(state, one_dimensional_action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_scores.append(score)
            global steps_done
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * steps_done / EPS_DECAY)
            episode_rand_prob.append(eps_threshold)
            plot_scores()
            plot_rand_prob()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()