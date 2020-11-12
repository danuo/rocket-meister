import gym, json
from ray.rllib import rollout
from ray.tune.registry import register_env

from rocket_gym import RocketMeister
class MultiEnv(gym.Env):
    def __init__(self, env_config):
        self.env = RocketMeister(env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)
    def render(self, mode):
        return self.env.render(mode)
register_env("rocketmeister", lambda c: MultiEnv(c))

# path to checkpoint
checkpoint_path = r'.\ray_results\Example\SAC_RocketMeister_ea992_00000_0_2020-11-11_22-07-33\checkpoint_3900\checkpoint-3900'

string = ' '.join([
    checkpoint_path,
    '--run',
    'SAC',
    '--env',
    'rocketmeister',
    '--episodes',
    '10',
    # '--no-render',
])

config = {
    'env_config': {
    # "export_frames": True,
    "export_states": True,
    'export_string': 'Training1', # filename prefix for exports
    },
}
config_json = json.dumps(config)
parser = rollout.create_parser()
args = parser.parse_args(string.split() + ['--config', config_json])

# ──────────────────────────────────────────────────────────────────────────
# if you want to automate this, by calling rollout.run() multiple times, you
# uncomment the following lines too. They need to called before calling
# rollout.run() a second, third, etc. time
# ray.shutdown()
# tune.register_env("rocketgame", lambda c: MultiEnv(c))
# from ray.rllib import _register_all
# _register_all()
# ──────────────────────────────────────────────────────────────────────────

rollout.run(args, parser)