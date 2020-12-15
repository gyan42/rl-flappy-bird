# rl_flappy_bird
Helping flappy bird to fly with Reinforcement Learning

## Python Environment Setup
```
conda create -n flappy_bird
pip install -r requirements.txt
``` 

## How to run?

1. Flappy Bird Python version

2. Train with Display
```
export PYTHONPATH=.
python fb/dqn.py --mode=train --display=true
```

3. Train without Display
On `GeForce GTX 1060 with Max-Q Design` it took ~ 1 hour to train on 1 lakh batches on average,
 with default settings to get trained!

```
export PYTHONPATH=.
python fb/dqn.py --mode=train --display=false
```

4. Test

```
export PYTHONPATH=.
# Model does pretty well after 4lakhs batch evaluations i.e 4 hours trianing...
python fb/dqn.py --mode=test --display=true --model_path=pretrained_model/current_model_400000.pth

# By default it uses `pretrained_model/latest_model.pth` -> `pretrained_model/current_model_1050000.pth`
python fb/dqn.py --mode=test --display=true

```

### References
- Theory: [https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial](https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial)
- [Bellman Equaltion](https://en.wikipedia.org/wiki/Bellman_equation)
- [Q-Learning Wiki](https://en.wikipedia.org/wiki/Q-learning)
- Python Flappy Bird port : [https://github.com/sourabhv/FlapPyBird](https://github.com/sourabhv/FlapPyBird)
- DQN: [https://github.com/nevenp/dqn_flappy_bird](https://github.com/nevenp/dqn_flappy_bird)
- Convolutional Network
    - Standford [CS231N](http://cs231n.stanford.edu/index.html)
    - [https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- Paper : [https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)


### Courses
- [https://www.davidsilver.uk/teaching/](https://www.davidsilver.uk/teaching/)