from W_Env.task_Goal_Action import task_Goal_Action
from W_Env.task_Temporal_Discounting import task_Temporal_Discounting
from W_Env.task_Horizon import task_Horizon
from W_Gym.W_Gym_simulator import W_env_simulator

class W_Env():
    env = None
    player = None
    envnames  = ["MC", "WV", "Horizon"]
    fullnames = ["task_Goal_Action", "task_Temporal_Discounting", "task_Horizon"]
    def __init__(self, envname, *arg, **kwarg):
        if not envname in self.envnames:
            raise Exception("env not defined")
        self.envname = envname
        if envname == "MC":
            self.env = task_Goal_Action(*arg, **kwarg)
        if envname == "WV":
            self.env = task_Temporal_Discounting(*arg, **kwarg)
        if envname == "Horizon":
            self.env = task_Horizon(*arg, **kwarg)

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
        if self.envname == "Horizon":
            player.set_keys(keys = ['space', 'left', 'right'], actions = [0,1,2])
        self.player = player
        return player

    def play(self):
        if self.player is None:
            self.get_player()
        self.player.play()

        