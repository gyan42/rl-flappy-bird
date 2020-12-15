import os

from fb.test import test
from fb.train import train
import sys
import time
import torch

from fb.nn.conv_net import NeuralNetwork
from fb.utils import init_weights

import fire


def main(mode,
         display,
         model_path=None):
    """
    Training Flappy Bird using RL
    :param mode: [train/test]
    :param display: [true/false]  Enable PyGame window
    :param model_path: Path to stored model path
    :return:
    """
    if display == 'true':
        from fb.env.flappy_bird import GameState
    else:
        os.putenv('SDL_VIDEODRIVER', 'fbcon')
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        from fb.env.flappy_bird import GameState

    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        game_state = GameState(FPS=30)
        if model_path:
            path = model_path
        else:
            path = 'pretrained_model/latest_model.pth'
        model = torch.load(
            path,
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model=model,
             game_state=game_state)

    elif mode == 'train':
        game_state = GameState(FPS=300)
        max_iteration = 2000000
        last_stored_iteration = 0
        if not os.path.exists('pretrained_model/latest_model.pth'):
            if not os.path.exists('pretrained_model/'):
                os.mkdir('pretrained_model/')
            model = NeuralNetwork(batch_size=256,
                                  number_of_actions=2,
                                  gamma=0.99,
                                  initial_epsilon=0.1,
                                  final_epsilon=0.0001,
                                  replay_memory_size=10000)
            last_stored_iteration = last_stored_iteration
        else:
            model = torch.load(
                'pretrained_model/latest_model.pth',
                map_location='cpu' if not cuda_is_available else None
            ).eval()
            model_files = os.listdir('pretrained_model/')
            model_files = [f for f in model_files if f.startswith("current_model_")]
            indexes = [int(index.replace("current_model_", "").replace(".pth", "")) for index in model_files]
            last_stored_iteration = max(indexes)

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model=model,
              game_state=game_state,
              max_iteration=max_iteration,
              iteration_index=last_stored_iteration,
              start=start)


if __name__ == "__main__":
    fire.Fire(main)