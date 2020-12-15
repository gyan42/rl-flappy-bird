import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from fb.nn.conv_net import NeuralNetwork
from fb.utils import resize_and_bgr2gray, image_to_tensor, init_weights


def train(model: NeuralNetwork,
          start,
          game_state,
          max_iteration=2000000,
          iteration_index=0):
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    # input_actions[0] == 1: do nothing
    # input_actions[1] == 1: flap the bird


    # initialize replay memory
    replay_memory = []

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)  # shape: (2,)
    action[0] = 1

    current_image_data, current_reward, current_terminal_state = game_state.frame_step(action)  # (288, 512, 3), float, bool

    current_image_data = resize_and_bgr2gray(current_image_data)  # (84, 84, 1)
    current_image_data = image_to_tensor(current_image_data)  # (1, 84, 84)
    # print("current_image_data", current_image_data.shape)

    # Conv net requires 4-dim image, hence concat B/W image four times
    # unsqueeze i.e inserts an dimension in specified axis : Eg: 84, 84 => 1, 84, 84
    # Reference:
    #       https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
    #       https://pytorch.org/docs/stable/generated/torch.cat.html
    current_state = torch.cat((current_image_data, current_image_data, current_image_data, current_image_data)).unsqueeze(0) # [1, 4, 84, 84]
    # print("current_state", current_state.shape) # [1, 4, 84, 84]
    # initialize epsilon value
    epsilon = model.initial_epsilon

    # Start with high learning rate and slow down as we train
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, max_iteration)

    pbar = tqdm(total=max_iteration + 1 - iteration_index)
    # main infinite loop
    while iteration_index < max_iteration:

        # get output from the neural network
        # start with initial screen shot
        output = model(current_state)[0]  # [2]

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)  # [2]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)]
        else:
            action_index = [torch.argmax(output)]

        action_index = action_index[0]


        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        # Is set either by the model output or by random action
        action[action_index] = 1

        # get next state and reward
        next_image_data, future_reward, future_terminal_state = game_state.frame_step(action)  # (288, 512, 3)

        next_image_data = resize_and_bgr2gray(next_image_data)  # (84, 84, 1)
        next_image_data = image_to_tensor(next_image_data)  # [1, 84, 84]

        # current_state # [1, 4, 84, 84]
        # print(current_state.squeeze(0).shape)   # [4, 84, 84]
        # print(current_state.squeeze(0)[1:, :, :].shape)   # [3, 84, 84]
        # Discard first channel in the image and add next screen shot to end of the image channel
        next_state = torch.cat((current_state.squeeze(0)[1:, :, :], next_image_data)).unsqueeze(0)  # [1, 4, 84, 84]
        future_reward = torch.from_numpy(np.array([future_reward], dtype=np.float32)).unsqueeze(0)  # [1, 1]

        action = action.unsqueeze(0)  # [1, 2]

        # save transition to replay memory
        replay_memory.append((current_state, action, future_reward, next_state, future_terminal_state))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration_index]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        current_state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        next_state_reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        next_state_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            current_state_batch = current_state_batch.cuda()  # [1, 4, 84, 84]
            action_batch = action_batch.cuda()  # [1, 2]
            next_state_reward_batch = next_state_reward_batch.cuda()  # [1, 1]
            next_state_batch = next_state_batch.cuda()  # [1, 4, 84, 84]

        # get output for the next state
        next_state_output = model(next_state_batch)  # [1, 2]

        # set y_j to r_j for terminal_state state,
        # otherwise to r_j + gamma * max(Q)
        # y_batch = torch.cat(tuple(next_state_reward_batch[i] if minibatch[i][4]
        #                           else next_state_reward_batch[i] + model.gamma * torch.max(next_state_output[i])
        #                           for i in range(len(minibatch))))
        y_batch = []
        for i in range(len(minibatch)):
            if minibatch[i][4]:
                res = next_state_reward_batch[i]
            else:
                res = next_state_reward_batch[i] + model.gamma * torch.max(next_state_output[i])   # [1]
                # print(res.shape)
            y_batch.append(res)

        y_batch = tuple(y_batch)
        y_batch = torch.cat(y_batch)  # [1]

        # extract Q-value
        q_value = torch.sum(model(current_state_batch) * action_batch, dim=1)  # [1]

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        current_state = next_state
        iteration_index += 1

        # if random_action:
        #     print("Performed random action!")

        if iteration_index % 25000 == 0:
            file = "pretrained_model/current_model_" + str(iteration_index) + ".pth"
            if os.path.exists(file):
                raise RuntimeError("Previous run models are found! Please remove it for a fresh run...")
            torch.save(model, file)
            torch.save(model, "pretrained_model/latest_model.pth")

        if iteration_index % 2500 == 0:

            print("iteration:", iteration_index, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
                  action_index.cpu().detach().numpy(), "reward:", future_reward.numpy()[0][0], "Q max:",
                  np.max(output.cpu().detach().numpy()))

        pbar.update(1)

    pbar.close()