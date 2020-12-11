# rl_flappy_bird
Helping flappy bird to fly with Reinforcement Learning

## Python Environment Setup
```
conda create -n flappy_bird
pip install -r requirements.txt
``` 

## Run steps

1. Flappy Bird Python version

2. Train
```
python fb/dqn.py train
```

3. Test
```
python fb/dqn.py test
```
### References
- Theory: [https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial](https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial)
- Python Flappy Bird port : [https://github.com/sourabhv/FlapPyBird](https://github.com/sourabhv/FlapPyBird)
- DQN: [https://github.com/nevenp/dqn_flappy_bird](https://github.com/nevenp/dqn_flappy_bird)
- Convolutional Network
    - Standford [CS231N](http://cs231n.stanford.edu/index.html)
    - [https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- Paper : [https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
