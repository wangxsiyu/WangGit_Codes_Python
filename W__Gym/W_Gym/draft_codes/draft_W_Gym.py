from collections import namedtuple
import pandas


class W_Gym_render(gym.Env):
    dt = None
    obs_Name2DimNumber = None
    plot_params = dict()
    currentscreen = None
    window = None
    clock = None
    font = None
    canvas = None
    text_T = None
    text_R = None
    image_lst = []
    metadata_render = {'render_mode': None, 'window_size': [512, 512], 'render_fps': None}
    def __init__(self, render_mode = None, window_size = [512, 512], \
                 render_fps = None, dt = 1,\
                 **kwarg):
        self.dt = dt
        metadata_render = W.W_dict_kwargs()
        W.W_dict_updateonly(self.metadata_render, metadata_render)
        if self.metadata_render['render_fps'] is None:
            self.metadata_render['render_fps'] = 1/self.dt
        self.metadata_render['window_size'] = np.array(self.metadata_render['window_size'])
        self.setup_rendermode()
        if hasattr(self, '_setup_render'):
            self._setup_render()
        # print(f"render mode: {self.metadata_render['render_mode']}")

    # draw obs
    def draw(self, channelname, val):
        if channelname == "ITI":
            return
        idx = self.obs_Name2DimNumber[channelname]
        if self.currentscreen is None:
            self.currentscreen = np.zeros(self.observation_space_size())
        self.currentscreen[idx] = val

    def observation_space_size(self): # deal with inconsistencies in gym output
        if self.observation_space.shape == ():
            return self.observation_space.n
        else:
            return self.observation_space.shape

    def _render_frame_1D(self, canvas, data, dictname):
        import pygame
        params = self.plot_params[dictname]
        n_channel = len(data)
        for ci in range(n_channel):
            tcol = params['colors'][ci]
            tplottype = params['plottypes'][ci]
            tradius = params['radius'][ci]
            tpos = None
            if params['position'] is not None:
                tpos = params['position'][ci]
            if tpos is None:
                tpos = np.array(self.window.get_size()) * [0.5,0.5]
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
        if self.plot_params['action']['plottypes'] == ['binary']:
            canvas = self._render_frame_binarychoice(canvas, np.array(self.plot_params['action']['plotparams']) == self.last_action)
        elif self.plot_params['action']['plottypes'] == ["arrows"]:
            canvas = self._render_frame_arrowchoice(canvas, np.array(self.plot_params['action']['plotparams']) == self.last_action)
        elif self.plot_params['action']['plottypes'] == ["arrowsplus"]:
            canvas = self._render_frame_arrowchoice(canvas, np.array(self.plot_params['action']['plotparams']) == self.last_action)
        else:
            canvas = self._render_frame_1D(canvas, W.enlist(self.last_action), 'action')
        return canvas

    def _render_frame_binarychoice(self, canvas, action):
        tradius = np.array(self.window.get_size()) * [0.05, 0.05]
        if action[0]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self.window.get_size()) * [0.1,0.5], tradius)
        if action[1]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self.window.get_size()) * [0.9,0.5], tradius)
        return canvas

    def _render_frame_arrowchoice(self, canvas, action):
        tradius = np.array(self.window.get_size()) * [0.05, 0.05]
        if action[0]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self.window.get_size()) * [0.1,0.5], tradius)
        if action[2]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self.window.get_size()) * [0.9,0.5], tradius)
        if action[1]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self.window.get_size()) * [0.5,0.1], tradius)
        if action[3]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self.window.get_size()) * [0.5,0.9], tradius)
        if len(action) > 4 and action[4]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self.window.get_size()) * [0.5,0.5], tradius)
        return canvas

    def _render_frame_obs(self, canvas, *arg, **kwarg):
        obs = list()
        lst = self.obs_Name2DimNumber
        for i, j in lst.items():
            obs.append(self.obs[np.array(j)])
        if hasattr(self, '_render_frame_obs_format'):
            obs = self._render_frame_obs_format(obs, lst)
        canvas = self._render_frame_1D(canvas, obs, 'obs')
        return canvas

    def flip(self, is_clear = True):
        self.obs = self.currentscreen
        if is_clear:
            self.blankscreen()

    def blankscreen(self):
        assert hasattr(self, 'observation_space')
        self.currentscreen = np.zeros(self.observation_space_size())

    def setup_rendermode(self, render_mode = None):
        if render_mode is None:
            render_mode = self.metadata_render['render_mode']
        else:
            self.metadata_render['render_mode'] = render_mode
        if render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.metadata_render['window_size'])
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font('freesansbold.ttf', 32)
            self.set_text('text_T', f"t = {0}")

    def setup_obs_Name2DimNumber(self, mydict):
        self.obs_Name2DimNumber = mydict

    def _render_setplotparams(self, dictname, plottypes = None, colors = None, radius = None, position = None, plotparams = None):
        params = W.W_dict_kwargs()
        del params['dictname']
        if dictname == "action" and position is None:
            position = np.array([0.1, 0.1])
            position = np.repeat(position[np.newaxis,:], len(params['plottypes']), axis = 0)
        for i in range(len(params['plottypes'])):
            if params['radius'] is not None:
                params['radius'][i] = params['radius'][i] * self.metadata_render['window_size']
            if params['position'] is not None and params['position'][i] is not None:
                params['position'][i] = params['position'][i] * self.metadata_render['window_size']
        self.plot_params.update({dictname: params})

    def set_text(self, attr, str):
        black = (0, 0, 0)
        red = (255, 0, 0)
        setattr(self, attr, self.font.render(str, True, red, black))


    def _render_array(self, z, pos, tradius = [1,1]):
        tradius = np.array(tradius)
        if len(z.shape) < 2:
            z = z.reshape(z.shape.__add__((1,)))
        if len(z.shape) == 2:
            z = np.stack([z,z,z], axis = 2)
        rx = np.ceil(tradius[0]/z.shape[0])
        ry = np.ceil(tradius[1]/z.shape[1])
        r = np.min((rx, ry))
        if r > 1:
            r = r.astype('int')
            z = np.kron(z, np.ones((r,r,1)))
        import pygame
        surf = pygame.surfarray.make_surface(z)
        image = {'image':surf, 'pos':pos - np.array(surf.get_size())/2}
        self.image_lst.append(image)

    def _render_frame_update(self):
        canvas = self.canvas
        if self.metadata_render['render_mode'] == "human":
            import pygame
            assert self.window is not None
            self.window.blit(canvas, canvas.get_rect())
            if self.image_lst != []:
                for x in self.image_lst:
                    self.window.blit(x['image'], x['pos'])
                self.image_lst = []
            if self.text_T is not None:
                trect = self.text_T.get_rect()
                wsize = pygame.display.get_window_size()
                trect.center = tuple(map(lambda i, j: i-j, wsize, trect.center))
                self.window.blit(self.text_T, trect)
            if self.text_R is not None:
                self.window.blit(self.text_R, self.text_R.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata_render["render_fps"])
        else:  # rgb_array or single_rgb_array
            import numpy as np
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _render_frame_create(self):
        import pygame
        canvas = pygame.Surface(self.metadata_render['window_size'])
        canvas.fill((0, 0, 0))
        return canvas

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

    def render(self, option = None, *arg, **kwarg):
        if self.metadata_render['render_mode'] is None:
            return
        self.canvas = self._render_frame_create()
        option = W.enlist(option)
        for x in option:
            self.render_frame(x, *arg, **kwarg)
        self._render_frame_update()

    def render_frame(self, option = None, *arg, **kwarg):
        canvas = self.canvas
        if option is None:
            assert hasattr(self, '_render_frame')
            self.canvas = self._render_frame(canvas, *arg, **kwarg)
        elif option == "action":
            self.canvas = self._render_frame_action(canvas, *arg, **kwarg)
        elif option == "obs":
            self.canvas = self._render_frame_obs(canvas, *arg, **kwarg)
        elif option == "reward":
            self._render_frame_reward(*arg, **kwarg)
        elif option == "time":
            self._render_frame_time(*arg, **kwarg)
        else:
            assert hasattr(self, '_render_frame')
            self.canvas = self._render_frame(canvas, option, *arg, **kwarg)

    def _render_frame_reward(self, *arg, **kwarg):
        if hasattr(self, 'last_reward'):
            self.set_text('text_R', f"R = {self.last_reward}")

    def _render_frame_time(self, *arg, **kwarg):
        self.set_text('text_T', f"game#{self.tot_trials}, { self.metadata_stage['stage_names'][self.stage] }, t = {self.t}")

    def get_window_relative2absolute(self, pos):
        import numpy as np
        pos = np.array(pos)
        return pos * self.metadata_render['window_size']

class W_Gym(W_Gym_render):
    obs = None
    obs_augment = None
    is_augment_obs = False
    is_faltten_obs = True
    info = None
    n_maxTrials = None
    tot_t = 0
    t = 0 # total time from beginning of a trial
    timer = 0 # timer
    stage = 0 # task stage
    tot_trials = 0 # total trials from beginning
    trial_counter = 0 # can be reset
    metadata_stage = {'stage_names':None, 'stage_timings': None, 'stage_advanceuponaction': None}
    last_reward = 0
    last_action = None
    last_stage = None
    is_oracle = False
    # action_immediateadvance = None
    def __init__(self, is_oracle = False):


        self.info = {'info_task':[], 'info_block':[], 'info_trial':[], 'info_step': []}
        self.setW_stage(["stages"], [np.Inf])

    def _len_observation(self):
        len = self.observation_space_size()
        if self.is_augment_obs:
            len += self._len_actions() + 1
        return len

    def _len_actions(self):
        return self.action_space.n

    # flow
    def reset(self):
        self.info['info_task'] = None

        self.last_action = None
        self.last_reward = 0

        self.render(option = ['obs', 'time'])

        self.task_info()
        self.info_step = {'obs':self._get_obs(False)}

    def reset_trial():
        last_trial = self.trial_counter


    def step_info(self):
        pass

    def _step_info(self, obs, action, reward, is_done, tot_t):
        self.info_step.update({'action':action, 'reward':reward, 'tot_t': tot_t, 'is_done':is_done, 'obs_next': obs})
        self.info_step.update({'blockID': self.tot_blocks, 'trialID': self.trial_counter, 't':self.t, 'stage': self.stage})
        self.step_info()
        self.info['info_step'].append(self.info_step)
        self.info_step = {'obs':obs}

    def block_info(self):
        self.info_block = {'blockID': self.tot_blocks}
        if hasattr(self, '_block_info'):
            self._block_info()
        self.info['info_block'].append(self.info_block)

    def trial_info(self, last_trial):
        self.info_trial = {'blockID': self.tot_blocks, 'trialID': last_trial}
        if hasattr(self, '_trial_info'):
            self._trial_info()
        if hasattr(self, '_get_oracle_trial'):
            self.info_trial['oracle'] = self._get_oracle_trial()
        self.info['info_trial'].append(self.info_trial)

    def task_info(self):
        self.info['info_task'] = self.info_task

    def get_oracle_action(self):
        return self.action_oracle.pop(0)

    def step(self, action):
        # record current choice
        self.last_stage = self.stage


        last_t = self.tot_t





        self.render(option = ["obs","action","reward","time"])

        if hasattr(self, '_get_oracle_step'):
            self._get_oracle_step()
        self._step_info(self._get_obs(False), action, self.last_reward, is_done, last_t)
        info = self._get_info()
        return obs, self.last_reward, is_done, self.tot_t, info


    def _advance_stage(self):
        is_error = False
        return self.stage + 1, is_error


    # action

    # get obs
    def _get_info(self):
        return self.info

    def _action_flattened(self):
        if not hasattr(self, '_action_transform'):
            action = np.zeros(self.action_space.n)
        else:
            assert hasattr(self, '_action_dimension')
            action = np.zeros(self._action_dimension)
        if self.last_action is not None:
            action[self.last_action] = 1
        return action

    def format4save(self):
        info = self._get_info()
        d1 = self.info2pandas(info['info_step'])
        d2 = self.info2pandas(info['info_trial'])
        d2 = pandas.concat([d2.drop(columns = 'params'), self.info2pandas(list(d2.params))], axis = 1)
        d3 = self.info2pandas(info['info_block'])
        d3 = pandas.concat([d3.drop(columns = 'params'), self.info2pandas(list(d3.params))], axis = 1)

        d23 = pandas.merge(d2, d3, on  = "blockID")

        data = pandas.merge(d1, d23, on = ["blockID", "trialID"])
        return data

    def info2pandas(self, tinfo):
        namestep = tuple(tinfo[0].keys())
        steptp = namedtuple('step', namestep)
        step = [steptp(*v.values()) for v in tinfo]
        step = steptp(*zip(*step))
        step = [list(np.stack(x)) for x in step]
        step = {k:v for k,v in zip(namestep, step)}
        data = pandas.DataFrame.from_dict({k:step[k] for k in list(set(step.keys()))})
        return data
