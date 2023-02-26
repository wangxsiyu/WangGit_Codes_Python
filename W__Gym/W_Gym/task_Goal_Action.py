from W_Gym_Grid2D import W_Gym_grid2D, grid2D
from gym import spaces
import numpy as np

class task_Goal_Action(W_Gym_grid2D):
    p_reward = None
    def __init__(self, p_reward = 1, **kwarg):
        super().__init__(3,3,7, **kwarg)
        self.p_reward = p_reward
        # set action space
        self.action_space = spaces.Discrete(5) # stay, left, up, right, down
        self.gaze = grid2D(0,2,0,2) # action object
        # set rendering dimension names
        self.setup_obs_Name2DimNumber({'fixation':0, \
                                       'square':1, 'imageA':2, 'imageB':3, \
                                        "dot1":4, "dot2":5, "reward":6})
        # set stages
        stage_names = ["fix0", "flash1", "post1", \
                       "flash2", "post2", "choice1", "hold1", \
                        "choice2", "hold2", "reward", "ITI"]
        stage_advanceuponaction = ["choice1", "choice2", "reward"]
        self.setW_stage(stage_names = stage_names, stage_advanceuponaction = stage_advanceuponaction)
        # task constants 
        self.cccs = np.array([6,8,2,0])
        self.ccs = np.array([3,7,5,1])
    
    def _setup_render(self):
        plottypes = ["circle", "square", "square", "square", "square", "square", "square"]
        colors = [(255,255,255), (255,255,255), (0, 255, 0), (0, 0, 255), (255,255,255), (255,255,255), (0, 255, 255)]
        radius = [0.02, 0.04, 0.08, 0.08, 0.04, 0.08, 0.12]
        self._render_setGrid2Dparams('obs', plottypes, colors, radius)
        plottypes = ["circle"]
        colors = [(255,0,0)]
        radius = [0.01]
        self._render_setGrid2Dparams('action', plottypes, colors, radius)
        
    def _render_frame_action(self, canvas):
        canvas = self._render_frame_grid2D(canvas, self.gaze.pos_grid2D, 'action')
        return canvas
    
    def _render_frame_obs(self, canvas):
        canvas = self._render_frame_grid2D(canvas, self.obs, 'obs')
        return canvas
    
    def _reset_block(self):
        self.oracle = np.random.randint(2)
    
    def _reset_trial(self):
        cccs = self.cccs 
        ccs = self.ccs 
        ccc_id = np.random.choice(4,2, replace = False)
        cc_id = np.array([0,2]) + np.random.choice(2,1)
        param = {'ccc_id': ccc_id, 'ccc_pos': cccs[ccc_id], 'cc_id':cc_id, 'cc_pos':ccs[cc_id]}
        self.param_trial = param
        self.gaze.set_pos(1,1)
        self.pos_choice1 = None
        self.pos_choice2 = None
    
    def get_valid_option3(self, action):
        action = np.where(action == self.ccs)[0][0]
        te = np.array([action, action-1])
        te[te == -1] = 3
        cccs = self.cccs
        return cccs[te].tolist()

    def vec2mat(self, gaze):
        y = gaze % 3
        x = int((gaze - y)/3)
        return x, y

    def mat2vec(self, x, y):
        gaze = x * 3 + y
        return gaze

    def action2dir(self, action):
        if action > 0:
            action = 5 - action
        if action % 2 == 0:
            dx = 0
        else:
            dx = action - 2
        if action % 2 == 1 or action == 0:
            dy = 0
        else:
            dy = 3 - action                        
        # mapping 
        # action:0, 1,  2,  3,  4
        # dx:    0, -1, 0,  1,  0
        # dy:    0, 0,  1,  0, -1
        return dx, dy
    
    def _step(self, action):
        R_int = 0
        R_ext = 0
        is_error = False
        is_done = False
        dx, dy = self.action2dir(action)
        gx, gy = self.gaze.move(dx, dy)
        gz = self.mat2vec(gx, gy)

        # need more work
        return R_ext, R_int, is_done