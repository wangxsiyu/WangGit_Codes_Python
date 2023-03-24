from W_Env.W_Env import W_Env_player

render_mode = "human"
n_maxTrials = 100
env = W_Env_player('TwoStep_Confidence_mini', render_mode = render_mode, \
                        n_maxTrials = n_maxTrials)
env.play()