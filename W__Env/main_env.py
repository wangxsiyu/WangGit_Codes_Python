from task_Goal_Action import task_Goal_Action
from task_Temporal_Discounting import task_Temporal_Discounting
from task_Horizon import task_Horizon
from W_Gym.W_Gym_simulator import W_env_simulator

render_mode = "human"
n_maxTrials = 100
env = task_Horizon(render_mode = render_mode, \
                        n_maxTrials = n_maxTrials)
player = W_env_simulator(env)
# player.set_keys(keys = ['space', 'left', 'up', 'right','down'], actions = [0,1,2,3,4])
player.set_keys(keys = ['fix', 'left', 'right'],actions = [0,1,2])
player.play("human")


