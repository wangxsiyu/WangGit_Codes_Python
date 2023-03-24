from task_TwoStep_Confidence_mini import task_TwoStep_Confidence_mini
from W_Gym.W_Gym_simulator import W_env_simulator
render_mode = "human"
n_maxTrials = 100
env = task_TwoStep_Confidence_mini(render_mode = render_mode, \
                        n_maxTrials = n_maxTrials, is_ITI = False)
player = W_env_simulator(env)
player.set_keys(keys = ['space', 'left', 'up', 'right','down'], actions = [4,0,2,1,3])
player.play()