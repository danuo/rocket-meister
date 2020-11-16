# RocketMeister
RocketMeister is an extensive and sophisticated gym environment for developing and comparing reinforcement learning algorithms. 
<p align="center">
  <img width="55%" src="https://raw.githubusercontent.com/danuo/rocket-meister/master/media/landing_gif.gif"><br>
  <a href="https://www.youtube.com/watch?v=GZlHsuTJG58">Link to YouTube video</a>
</p>

There is also a writeup on Medium aimed towards people that are new to reinforcement learning. The articles explain many of the ideas and concepts behind the decisions made during the creation of the environment. You can find the articles here: [Medium article part 1](https://medium.com/@d.brummerloh/ultimate-guide-for-reinforced-learning-part-1-creating-a-game-956f1f2b0a91), [Medium article part 2](https://medium.com/@d.brummerloh/ultimate-guide-for-ai-game-creation-part-2-training-e252108dfbd1)


```python3
from rocket_gym import RocketMeister10 
env = RocketMeister10(env_config={'keyword': value})
```
#### Requirements
```
gym 
pygame 2.0.0           (for playing and rendering)
ray 1.0                (for training and rollout)
tensorflow OR pytorch  (for training and rollout)
```
#### These are the key features:
* **Gym environment with various settings.** For detailed configuration options, see below.
* **Playable by humans.** Rendering and interaction is implemented through `pygame` library and can be accessed by the script `start_human.py`.
* **Policy training with various reinforcement learning algorithms** The environment can be trained with various agents, such as SAC, PPO and ARS. Agent training is implemented with the `ray` library and can be accessed by the script `start_training.py`.
* **Export of frames and replays** The environment can export rendered frames with the `export_frames` environment keyword. Replays can be exported with export_states flag. Multiple replays can later be rendered to create a video. To render multiple replays, use the script `start_replay_renderer.py`.
* **Level Generator** The environment features a level generator to prevent overfitting during training. One can achieve a high level of generalizsation with the trained policy. Use the level generator to generate random levels by setting `env_name` to 'random' inside the `env_config`.
<p align="center">
  <img width="55%" src="https://raw.githubusercontent.com/danuo/rocket-meister/master/media/levelgen_gif.gif"><br>
  Sample levels from the level generator.
</p>

### Play as human
**run** `start_human.py`  Try to achieve a good score yourself by playing the environment interactively. **The rocket is controlled with the arrow keys**, you can **reset the environment by pressing r**
* Export the replay of each round by passing `export_states: True` in the environment config. Later, you can render the replay with `start_replay_renderer.py`.
* Export each rendered frame as a .jpg file by passing `export_frames: True` in the environment config.

### Train a policy
**run** `start_ray_training.py`  This file allows you to train the environment with different reinforcement learning agents. To change the utilized agent, simply change the agent string from 'SAC' accordingly (for example 'PPO', 'DDPG' or 'ARS'). To see a full list of supported agents, visit the ray documentation: https://docs.ray.io/en/latest/rllib-algorithms.html

### Rollout a policy
**run** `start_ray_rollout.py`  This file allows you to rollout a previously trained policy. To do so, you need to set chechpoint_path to the checkpoint you want to use for the rollout.
* Export the replay of each round by passing `export_states: True` in the environment config. Later, you can render the replay with `start_replay_renderer.py`.
* Export each rendered frame as a .jpg file by passing `export_frames: True` in the environment config.

### Render replays
**run** `start_replay_renderer.py`  This file allows you to render multiple replays at once. 
* Export each rendered frame as a .jpg file by passing `export_frames: True` in the environment config.

# The RocketMeister environment
This segment will discuss the most important options offered by the environment. A full list of valid keywords and values can be found in the definition of the `parse_env_config()` function inside `rocket_gym.py`.

### Levels
At this point, RocketMeister features two levels and a level generator to allow the training of a more generalized policy. The levels can be accessed through the `env` keyword. Additionally, each level can be flipped (as in mirrored) with the `env_flipped` keyword to test against overfitting.
```python3
env_config = {
    'env': 'level1',
    'env_flipped': False,
    } 
from rocket_gym import RocketMeister10 
env = RocketMeister10(env_config)
```
Currently, these options are available for `env_name`:
* 'level1' (level 1)
* 'level2' (level 2)
* 'empty'  (no level)
* 'random' (randomly generated level)
<p align="center">
    <table style="width:60%">
        <tr>
            <th>
                <img width="85%" src="https://raw.githubusercontent.com/danuo/rocket-meister/master/media/env_level1.jpg"><br>
                env_name = 'level1' 
            </th>
            <th>
                <img width="85%" src="https://raw.githubusercontent.com/danuo/rocket-meister/master/media/env_level2.jpg"><br>
                env_name = 'level2'
                </th>
        </tr>
        <tr>
            <th>
                <img width="85%" src="https://raw.githubusercontent.com/danuo/rocket-meister/master/media/env_level1_flipped.jpg"><br>
                env_name = 'level1', env_flipped = True
                </th>
            <th>
                <img width="85%" src="https://raw.githubusercontent.com/danuo/rocket-meister/master/media/levelgen_gif.gif"><br>
                env_name = 'random'
                </th>
        </tr>
    </table>
</p>

### Observations
The RocketMeister environment features different sets of observations. Each set of observation is tied to a specific subclass of the environment, which all can be loaded with the following commands:
![](https://raw.githubusercontent.com/danuo/rocket-meister/master/media/formula_observations_all.png | width=50)
```python3
import rocket_gym
env = rocket_gym.RocketMeister10(env_config)
env = rocket_gym.RocketMeister9(env_config)
env = rocket_gym.RocketMeister8(env_config)
env = rocket_gym.RocketMeister7(env_config)
```
The individual observations are explained in the images below
<p align="center">
    <table style="width:60%">
        <tr>
            <th>
                <img width="85%" src="https://raw.githubusercontent.com/danuo/rocket-meister/master/media/echo.jpg"><br>
                Distances: These are the length of the 7 echo vectors seen in the picture.
            </th>
            <th>
                <img width="85%" src="https://raw.githubusercontent.com/danuo/rocket-meister/master/media/velocity.jpg"><br>
                Velocity: This is the absolute, non-directional velocity of the rocket.
            </th>
        </tr>
        <tr>
            <th>
                <img width="85%" src="https://raw.githubusercontent.com/danuo/rocket-meister/master/media/vel_ang.jpg"><br>
                Velocity angle: This angle defines the angle between the rocket's orientation and the rocket's velocity vector. Can become negative.
            </th>
            <th>
                <img width="85%" src="https://raw.githubusercontent.com/danuo/rocket-meister/master/media/goal_ang.jpg"><br>
                Goal angle: This angle defines the angle between the rocket's orientation and the vector perpendicular to the next goal. Can become negative.
            </th>
        </tr>
    </table>
</p>

