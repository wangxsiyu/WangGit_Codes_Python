import W_Python_Codes.W_tools as W_tools
import numpy as np
import gym
from gym import spaces
from gym.spaces import utils

class W_gym(gym.Env):
    metadata = {"render_mode": None, "render_fps": None, "dt": 100}
    R_default = {'R_advance', 0, 'R_error', -1}
    def __init__(self, render_mode = None, window_size = (512, 512), dt = 100, render_fps = None):
        metadata = W_tools.W_dict_kwargs()
        self.metadata = self.metadata.update(metadata)
        self.dt = self.metadata['dt']
        if self.metadata['render_fps'] is None:
            self.metadata['render_fps'] = 1000/self.dt
        self.setup_rendermode(self.metadata['render_mode'])
    
    def reset(self, seed = None, return_info = False):
        super().reset(seed = seed)
        if self.render_mode == "human":
            self.render()
        self.reward = 0 # total reward
        self.reward_external = 0 # actual reward that the animal receives
        self.reward_internal = 0 # shaping reward
        obs = self._get_obs()
        info = self._get_info()
        return obs if not return_info else (obs, info)

    def reset_trial(self):
        self.t = 0
        self.timer = 0
        self.trialID = self.trialID + 1
        self.release_hold()
        self.obs = np.zeros(self.observation_space.shape)

    def reset_block(self):
        self.trialID = 0

    def release_hold(self):
        self.action_to_hold = None
        
    def check_matchaction(self, action, action_correct):
        if action == action_correct:
            is_error = False
        else:
            is_error = True
        return is_error

    def check_holdaction(self, action, valid_actions = None):
        if self.action_to_hold is None:
            is_error = False
            if valid_actions is None or action in valid_actions:
                self.action_to_hold = action
        else:
            if action == self.action_to_hold:
                is_error = False
            else:
                is_error = True
        return is_error    

    def step(self):
        self.t = self.t + self.dt
        self.timer = self.timer + self.dt
    
    def setup_rendermode(self, render_mode = None):
        self.metadata['render_mode'] = render_mode
        if self.metadata['render_mode'] == "human":
            import pygame  # import here to avoid pygame dependency with no render
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.metadata['window_size'])
            self.clock = pygame.time.Clock()
        else:
            self.window = None
            self.clock = None

    def render(self):
        return self._render_frame(self.metadata['render_mode'])

    def _render_frame_create(self):
        import pygame 
        canvas = pygame.Surface(self.metadata['window_size'])
        canvas.fill((0, 0, 0))
        return canvas

    def _render_frame_update(self, canvas, mode: str):
        assert mode is not None
        if mode == "human":
            import pygame
            assert self.window is not None
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array or single_rgb_array
            import numpy as np
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def window_relativesize(self, pos):
        import numpy as np
        pos = np.array(pos)
        return pos * self.metadata['window_size']

    def close(self):
        if self.window is not None:
            import pygame 
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self):
        return self.obs

    def _get_info(self): # needs to return more things
        return {'total-reward': self.reward}


class grid2D():
    def __init__(self, x1 = -np.Inf, x2 = np.Inf, y1 = -np.Inf, y2 = np.Inf, x0 = None, y0 = None):
        self.xy_range = np.array([[x1, x2],[y1, y2]])
        self.set_pos(x0, y0)

    def set_pos(self, x0, y0):
        if x0 is not None and y0 is not None:
            self.pos_grid2D = np.array([x0, y0])
    
    def move(self, dx, dy, option = None, **kwarg):
        d = np.array([dx, dy])
        pos = d + self.pos_grid2D
        self.pos_grid2D = self.restrict2range(pos)
        return self.get_gaze(option, kwarg)

    def restrict2range(self, pos):
        for i in range(2):
            if pos[i] < self.xy_range[i][0]:
                pos[i] = self.xy_range[i][0]
            if pos[i] > self.xy_range[i][1]:
                pos[i] = self.xy_range[i][1]
        return 
    
    def get_gaze(self, option = "xy", myspace = None):
        x, y = (self.pos_grid2D[0], self.pos_grid2D[1])
        if option is None or option == "xy":
            return x, y
        elif option == "space":
            obs = np.zeros(option.shape)
            obs[x, y] = 1
            return utils.flatten(myspace, obs)

class W_gym_grid2D(W_gym):
    def __init__(self, nx, ny, ndim_obs, **kwarg):
        super().__init__(kwarg)
        self.observation_space = spaces.Box(low = 0, high = 1, shape = (nx, ny, ndim_obs), dtype = np.int8)
        self.obs = np.zeros(self.observation_space.shape)
        self.plotparams = {'obs':None, 'action':None}
        self.position = self.pos_grid(nx, ny) * self.metadata['window_size']

    def pos_grid(self, nx, ny):
        x = np.linspace(0, 1, nx, endpoint=True) + 1/nx/2
        y = np.linspace(0, 1, ny, endpoint=True) + 1/ny/2
        xv, yv = np.meshgrid(x, y)
        pos = np.array([(i, j) for i, j in zip(xv, yv)])
        return pos

    def blankscreen(self):
        self.currentscreen = np.zeros(self.observation_space.shape)
        
    def draw(self, x, y, channel):
        channelID = np.where(channel == self.map_item2dim)[0]
        self.currentscreen[x, y, channelID] = 1

    def flip(self, is_clear = True):
        obs = self.currentscreen
        if is_clear:
            self.blankscreen()
        return obs
    
    def setup_map_item2dim(self, **kwarg):
        self.map_item2dim = W_tools.W_dict_kwargs()

    def show_item_xy(self, obs, itemname, x, y):
        obs[x, y, self.map_item2dim[itemname]] = 1
        return obs

    def _render_frame_setparams(self, obs_or_act, plottypes = None, colors = None, radius = None):
        params = W_tools.W_dict_kwargs()
        for i in range(len(params['plottypes'])):
            tradius = params['radius'][i] * self.metadata['window_size']
            if params['plottypes'][i] == "circle":
                params['radius'][i] *= np.mean(tradius)
            elif params['plottypes'][i] == "rect":
                params['radius'][i] = tradius
        self.plotparams[obs_or_act] = params

    def _render_frame_grid2D(self, canvas, data, obs_or_act):
        import pygame
        params = self.plotparams[obs_or_act]
        n_channel = data.shape[2]            
        for ci in range(n_channel):
            tcol = params['colors'][ci]
            tplottype = params['plottypes'][ci]
            tradius = params['radius'][ci]
            for xi in range(self.observation_space.shape[0]):
                for yi in range(self.observation_space.shape[1]):
                    tval = data[xi, yi, ci]
                    tpos = self.position[xi, yi]
                    if tval > 0: # show
                        if tplottype == "circle":
                            pygame.draw.circle(canvas, tcol, tpos, np.mean(tradius))
                        elif tplottype == "rect":
                            pygame.draw.rect(canvas, tcol, 
                                np.concatenate((-tradius + tpos, tradius * 2), axis = None), 0)
