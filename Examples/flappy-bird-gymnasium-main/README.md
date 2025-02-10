# Flappy Bird for Gymnasium

![Python versions](https://img.shields.io/pypi/pyversions/flappy-bird-gymnasium)
[![PyPI](https://img.shields.io/pypi/v/flappy-bird-gymnasium)](https://pypi.org/project/flappy-bird-gymnasium/)
[![License](https://img.shields.io/github/license/markub3327/flappy-bird-gymnasium)](https://github.com/markub3327/flappy-bird-gymnasium/blob/master/LICENSE)

This repository contains the implementation of Gymnasium environment for
the Flappy Bird game. The implementation of the game's logic and graphics was
based on the [flappy-bird-gym](https://github.com/Talendar/flappy-bird-gym) project, by
[@Talendar](https://github.com/Talendar). 

## State space

The "FlappyBird-v0" environment, yields simple numerical information about the game's state as
observations representing the game's screen.

### `FlappyBird-v0`
There exist two options for the observations:  
1. option
* The LIDAR sensor 180 readings (Paper: [Playing Flappy Bird Based on Motion Recognition Using a Transformer Model and LIDAR Sensor](https://www.mdpi.com/1424-8220/24/6/1905))

2. option
* the last pipe's horizontal position
* the last top pipe's vertical position
* the last bottom pipe's vertical position
* the next pipe's horizontal position
* the next top pipe's vertical position
* the next bottom pipe's vertical position
* the next next pipe's horizontal position
* the next next top pipe's vertical position
* the next next bottom pipe's vertical position
* player's vertical position
* player's vertical velocity
* player's rotation

## Action space

* 0 - **do nothing**
* 1 - **flap**

## Rewards

* +0.1 - **every frame it stays alive**
* +1.0 - **successfully passing a pipe**
* -1.0 - **dying**
* −0.5 - **touch the top of the screen**

<br>

<p align="center">
  <img align="center" 
       src="https://github.com/markub3327/flappy-bird-gymnasium/blob/main/imgs/dqn.gif?raw=true" 
       width="200"/>
</p>

## Installation

To install `flappy-bird-gymnasium`, simply run the following command:

    $ pip install flappy-bird-gymnasium
    
## Usage

Like with other `gymnasium` environments, it's very easy to use `flappy-bird-gymnasium`.
Simply import the package and create the environment with the `make` function.
Take a look at the sample code below:

```python
import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()
```

## Playing

To play the game (human mode), run the following command:

    $ flappy_bird_gymnasium
    
To see a random agent playing, add an argument to the command:

    $ flappy_bird_gymnasium --mode random

To see a Deep Q Network agent playing, add an argument to the command:

    $ flappy_bird_gymnasium --mode dqn
