from W_Env.W_Env import W_Env

render_mode = "human"
n_maxTrials = 100
env = W_Env('TwoStep', render_mode = render_mode, \
                        n_maxTrials = n_maxTrials, \
                        is_save = True)
env.play(mode = "human")