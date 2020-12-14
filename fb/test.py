import torch
from fb.nn.conv_net import NeuralNetwork
from fb.utils import resize_and_bgr2gray, image_to_tensor

def test(model: NeuralNetwork,
         game_state):

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal_state = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal_state = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1
