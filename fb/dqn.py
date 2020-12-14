import os

from fb.test import test
from fb.train import train
import sys
import time
import torch

from fb.nn.conv_net import NeuralNetwork
from fb.utils import init_weights

def main(mode, display):
    if display == 'true':
        from fb.env.flappy_bird import GameState
    else:
        os.putenv('SDL_VIDEODRIVER', 'fbcon')
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        from fb.env.flappy_bird import GameState

    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        game_state = GameState(FPS=30)
        model = torch.load(
            'pretrained_model/latest_model.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model,
             game_state=game_state)

    elif mode == 'train':
        game_state = GameState(FPS=30)
        max_iteration = 2000000
        last_stored_iteration = -1
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')
            model = NeuralNetwork(batch_size=256,
                                  number_of_actions=2,
                                  gamma=0.99,
                                  initial_epsilon=0.1,
                                  final_epsilon=0.0001,
                                  replay_memory_size=10000)
            last_stored_iteration = 0
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

        train(model,
              game_state=game_state,
              max_iteration=max_iteration,
              iteration_index=last_stored_iteration,
              start=start)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])