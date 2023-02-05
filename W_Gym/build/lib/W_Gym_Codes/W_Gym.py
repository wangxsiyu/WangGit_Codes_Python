import gymnasium as gym
from gymnasium import spaces

class W_gym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    def __init__(self, render_mode = None, window_size = (512, 512), render_fps = 20, dt = None):
        self.metadata['render_modes'] = render_mode
    