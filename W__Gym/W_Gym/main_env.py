from task_Goal_Action import task_Goal_Action
from W_Gym_simulator import W_env_simulator
render_mode = "human"
n_maxTrialsPerBlock = 100
n_maxTrials = 100
env = task_Goal_Action(render_mode = render_mode, \
                        n_maxTrialsPerBlock = n_maxTrialsPerBlock, \
                        n_maxTrials = n_maxTrials)
player = W_env_simulator(env)
player.set_keys(keys = ['space', 'left', 'right', 'up', 'down'], actions = [0,1,2,3,4])
player.play("human")


