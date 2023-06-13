import torch

class W_Worker:
    def __init__(self, env, model, recording_mode = "state-action", *arg, **kwarg):
        self.env = env
        self.model = model
        self.recording_mode = recording_mode # state-action, behavior, neurons 
       
    def run_episode(self, device = "cpu", mode_action = "softmax"):
            model = self.model
            env = self.env
            recording_mode = self.recording_mode
            assert model is not None
            assert env is not None

            done = False
            total_reward = 0
            if recording_mode in ["behavior", "neurons"]:
                env.saveon()
            if recording_mode == "neurons":
                recording_neurons = []
            obs = env.reset()

            mem_state = None
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

            if recording_mode == "neurons":
                return (env._data, recording_neurons)
            elif recording_mode == "behavior":
                return env._data
            elif recording_mode == "state-action":
                return 
            return env._data, recording_neurons if record else total_reward