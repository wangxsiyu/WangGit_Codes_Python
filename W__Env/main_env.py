from task_Goal_Action import task_Goal_Action
from task_Temporal_Discounting import task_Temporal_Discounting
from W_Gym.W_Gym_simulator import W_env_simulator
render_mode = "human"
n_maxTrials = 100
env = task_Temporal_Discounting(render_mode = render_mode, \
                        n_maxTrials = n_maxTrials)
player = W_env_simulator(env)
# player.set_keys(keys = ['space', 'left', 'up', 'right','down'], actions = [0,1,2,3,4])
player.set_keys(keys = ['space','a'],actions = [0,1])
player.play("human")


