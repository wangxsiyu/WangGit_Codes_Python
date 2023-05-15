from W_Gym.W_Gym_task import W_Gym_task
from W_Python import W_tools as W
import numpy as np

class W_Gym_renderer(W_Gym_task):
    metadata_render = {'render_mode': None, 'window_size': [512, 512], 'render_fps': None}
    _renderer_window = None
    _renderer_clock = None
    _renderer_canvas = None
    _renderer_font = None
    _renderer_text_T = None
    _renderer_text_R = None
    # objects to render
    _renderer_auto_params = {}
    _renderer_object_array = []
    # interactive component
    human_key_action = None

    def __init__(self, render_mode = None, window_size = [512, 512], \
                 render_fps = None, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        metadata_render = W.W_dict_kwargs()
        W.W_dict_updateonly(self.metadata_render, metadata_render)
        if self.metadata_render['render_fps'] is None:
            self.metadata_render['render_fps'] = 1/self._time_unit
        self.metadata_render['window_size'] = np.array(self.metadata_render['window_size'])
        self.setup_rendermode()
        if hasattr(self, 'setup_render_parameters'):
            self.setup_render_parameters()

    def setup_rendermode(self, render_mode = None):
        if render_mode is None:
            render_mode = self.metadata_render['render_mode']
        else:
            self.metadata_render['render_mode'] = render_mode
        if render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render
            pygame.init()
            pygame.display.init()
            self._renderer_window = pygame.display.set_mode(self.metadata_render['window_size'])
            self._renderer_clock = pygame.time.Clock()
            self._renderer_font = pygame.font.Font('freesansbold.ttf', 32)
            self._render_text('_renderer_text_T', f"t = {0}")

    def render(self, option = None, *arg, **kwarg):
        if self.metadata_render['render_mode'] is None:
            return
        self._renderer_canvas = self._render_frame_create()
        canvas = self._renderer_canvas
        option = W.enlist(option)
        for x in option:    
            if x is None:
                assert hasattr(self, 'custom_render_frame')
                self._renderer_canvas = self.custom_render_frame(canvas, *arg, **kwarg)
            elif x == "action":
                self._renderer_canvas = self._render_frame_action(canvas, *arg, **kwarg)
            elif x == "obs":
                self._renderer_canvas = self._render_frame_obs(canvas, *arg, **kwarg)
            elif x == "reward":
                self._render_frame_reward(*arg, **kwarg)
            elif x == "time":
                self._render_frame_time(*arg, **kwarg)
            else:
                assert hasattr(self, 'custom_render_frame')
                self._renderer_canvas = self.custom_render_frame(canvas, x, *arg, **kwarg)
        return self._render_frame_update()

    def _render_text(self, attr, str):
        black = (0, 0, 0)
        red = (255, 0, 0)
        setattr(self, attr, self._renderer_font.render(str, True, red, black))

    def _render_frame_create(self):
        import pygame
        _renderer_canvas = pygame.Surface(self.metadata_render['window_size'])
        _renderer_canvas.fill((0, 0, 0))
        return _renderer_canvas
    
    def _render_frame_update(self):
        canvas = self._renderer_canvas
        if self.metadata_render['render_mode'] == "human":
            import pygame
            assert self._renderer_window is not None
            self._renderer_window.blit(canvas, canvas.get_rect())
            if self._renderer_object_array != []:
                for x in self._renderer_object_array:
                    self._renderer_window.blit(x['image'], x['pos'])
                self._renderer_object_array = []
            if self._renderer_text_T is not None:
                trect = self._renderer_text_T.get_rect()
                wsize = pygame.display.get_window_size()
                trect.center = tuple(map(lambda i, j: i-j, wsize, trect.center))
                self._renderer_window.blit(self._renderer_text_T, trect)
            if self._renderer_text_R is not None:
                self._renderer_window.blit(self._renderer_text_R, self._renderer_text_R.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self._renderer_clock.tick(self.metadata_render["render_fps"])
            return None
        elif self.metadata_render['render_mode'] == "rgb_array":  # rgb_array or single_rgb_array
            import numpy as np
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _render_frame_reward(self, *arg, **kwarg):
        if hasattr(self, '_last_reward'):
            self._render_text('_renderer_text_R', f"R = {self._last_reward}")
    
    def _render_frame_time(self, *arg, **kwarg):
        self._render_text('_renderer_text_T', f"game#{self._count_task_trial}, " + \
                        f"{self._param_state['names'][self._state]}, " + \
                        f"t = {self._time_trial}")

    def get_pos_relative2absolute(self, pos):
        import numpy as np
        pos = np.array(pos)
        return pos * self.metadata_render['window_size']
    
    def render_close(self):
        if self._renderer_window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

    def setup_human_keys_auto(self, option = None):
        if option == "binary":
            keys = ['left', 'right']
            actions = [0,1]
        elif option == "binary_plus":
            keys = ['space', 'left', 'right']
            actions = [0,1,2]
        elif option == "arrows":
            keys = ['left', 'up', 'right', 'down']
            actions = [0,1,2,3]
        elif option == "arrows_plus":
            keys = ['space', 'left', 'up', 'right','down']
            actions = [0,1,2,3,4]
        self.set_human_keys(keys, actions)

    def set_human_keys(self, keys, actions):
        self.human_key_action = {'keys':keys, 'actions': actions}

    def _render_set_auto_parameters(self, plotname, plottypes = None, colors = None, radius = None, position = None, additional_params = None):
        params = W.W_dict_kwargs()
        del params['plotname']
        # change relative size to absolute size
        for i in range(len(params['plottypes'])):
            if params['radius'] is not None:
                params['radius'][i] = self.get_pos_relative2absolute(params['radius'][i])
            if params['position'] is not None and params['position'][i] is not None:
                params['position'][i] = self.get_pos_relative2absolute(params['position'][i])
        self._renderer_auto_params.update({plotname: params})

    def _render_frame_obs(self, canvas, *arg, **kwarg):
        obs = list()
        lst = self._obs_channelID
        obs = self._obs
        if hasattr(self, 'custom_render_frame_obs_format'):
            obs = self.custom_render_frame_obs_format(obs, lst)
        if obs is not None:
            canvas = self._render_frame_autodraw(canvas, obs, self._renderer_auto_params['obs'])
        return canvas
            
    def _render_frame_autodraw(self, canvas, data, params):
        import pygame
        n_channel = len(data)
        for ci in range(n_channel):
            tcol = params['colors'][ci]
            tplottype = params['plottypes'][ci]
            tradius = params['radius'][ci]
            tpos = None
            if params['position'] is not None:
                tpos = params['position'][ci]
            if tpos is None:
                tpos = np.array(self._renderer_window.get_size()) * [0.5,0.5]
            tval = data[ci]
            if np.any(tval > 0): # show
                canvas = self._render_draw(canvas, tplottype, tcol, tpos, tradius, tval)
        return canvas
    
    def _render_draw(self, canvas, tplottype = None, tcol = None, tpos = None, tradius = None, tval = None):
        import pygame
        if tplottype == "circle":
            pygame.draw.circle(canvas, tcol, tpos, np.mean(tradius))
        elif tplottype == "square":
            pygame.draw.rect(canvas, tcol, \
                np.concatenate((-tradius + tpos, tradius * 2), axis = None), 0)
        elif tplottype == "image":
            self._render_array(tval, tpos, tradius)
        return canvas

    def _render_frame_action(self, canvas, *arg, **kwarg):
        if self._renderer_auto_params['action']['plottypes'] == ['action_binary']:
            canvas = self._render_frame_binarychoice(canvas, np.array(self._renderer_auto_params['action']['additional_params']) == self._last_action)
        elif self._renderer_auto_params['action']['plottypes'] == ["action_arrows"]:
            canvas = self._render_frame_arrowchoice(canvas, np.array(self._renderer_auto_params['action']['additional_params']) == self._last_action)
        elif self._renderer_auto_params['action']['plottypes'] == ["action_arrowsplus"]:
            canvas = self._render_frame_arrowchoice(canvas, np.array(self._renderer_auto_params['action']['additional_params']) == self._last_action)
        else:
            canvas = self._render_frame_autodraw(canvas, W.enlist(self._last_action), 'action')
        return canvas

    def _render_frame_binarychoice(self, canvas, action):
        tradius = np.array(self._renderer_window.get_size()) * [0.05, 0.05]
        if action[0]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self._renderer_window.get_size()) * [0.1,0.5], tradius)
        if action[1]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self._renderer_window.get_size()) * [0.9,0.5], tradius)
        return canvas
