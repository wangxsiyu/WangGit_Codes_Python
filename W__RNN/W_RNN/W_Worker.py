import torch

class W_Worker:
    def __init__(self, env, model, recording_mode = "state_action", *arg, **kwarg):
        self.env = env
        self.model = model
        self.recording_mode = recording_mode # state_action, behaviors, neurons 
       
    def run_episode(self, device = "cpu", mode_action = "softmax"):
            model = self.model
            env = self.env
            recording_mode = self.recording_mode
            assert model is not None
            assert env is not None

            done = False
            total_reward = 0
            if isrecord:
                env.saveon()
            obs = env.reset()

            mem_state = None
            if isrecord:
                recording_neurons = []
            while not done:
                # take actions
                obs = torch.from_numpy(obs).unsqueeze(0).float()
                action_vector, val_estimate, mem_state = model(obs.unsqueeze(0).to(device), mem_state)
                if record:
                    recording_neurons.append(mem_state[0])
                action = self.select_action(action_vector, mode_action)
                obs_new, reward, done, timestep = env.step(action.item())
                reward = float(reward)
                obs = obs_new
                total_reward += reward
            if record:
                recording_neurons = torch.concat(recording_neurons).squeeze()
            return env._data, recording_neurons if record else total_reward