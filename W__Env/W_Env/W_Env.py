from W_Env.task_Goal_Action import task_Goal_Action
from W_Env.task_Temporal_Discounting import task_Temporal_Discounting
from W_Env.task_Horizon import task_Horizon
from W_Env.task_TwoStep import task_TwoStep
from W_Env.task_TwoStep_Confidence import task_TwoStep_Confidence
from W_Env.task_TwoStep_simple import task_TwoStep_simple
from W_Gym.W_Gym_simulator import W_env_simulator

def W_Env(envname, *arg, **kwarg):
    envnames  = ["MC", "WV", "Horizon", "TwoStep", "TwoStep_Confidence","TwoStep_simple"]
    fullnames = ["task_Goal_Action", "task_Temporal_Discounting", "task_Horizon", 'task_TwoStep', 'task_TwoStep_Confidence']
    if not envname in envnames:
        raise Exception("env not defined")
    if envname == "MC":
        env = task_Goal_Action(*arg, **kwarg)
    if envname == "WV":
        env = task_Temporal_Discounting(*arg, **kwarg)
    if envname == "Horizon":
        env = task_Horizon(*arg, **kwarg)
    if envname == "TwoStep":
        env = task_TwoStep(*arg, **kwarg)    
    if envname == "TwoStep_Confidence":
        env = task_TwoStep_Confidence(*arg, **kwarg)
    if envname == "TwoStep_simple":
        env = task_TwoStep_simple(*arg, **kwarg)
    return env

class W_Env_player():
    env = None
    player = None
    def __init__(self, envname, *arg, **kwarg):
        self.env = W_Env(envname, *arg, **kwarg)
        self.envname = envname

    def get_env(self):
        return self.env
    
    def get_player(self):
        if self.player is not None:
            return self.player
        player = W_env_simulator(self.env)
        if self.envname == "MC":
            player.set_keys(keys = ['space', 'left', 'up', 'right','down'], actions = [0,1,2,3,4])
        if self.envname == "WV":
            player.set_keys(keys = ['space', 'a'], actions = [0,1])
        if self.envname in ["Horizon", "TwoStep", "TwoStep_simple"]:
            player.set_keys(keys = ['space', 'left', 'right'], actions = [0,1,2])
        if self.envname in ["TwoStep_Confidence"]:
            player.set_keys(keys = ['space', 'left', 'right', 'up'], actions = [0,1,2,3])
        self.player = player
        return player

    def play(self):
        if self.player is None:
            self.get_player()
        self.player.play()