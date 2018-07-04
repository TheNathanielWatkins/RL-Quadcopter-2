import numpy as np
# from task import Task

class PolicySearch_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.score = 0.
        self.average_score = 0.
        self.total_score = []
        self.noise_scale = 0.1
        self.params = "noise_scale: {}".format(self.noise_scale)

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.score = 0.
        state = self.task.reset()
        return state

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.score += reward

        # Learn, if at end of episode
        if done:
            self.total_score.append(self.score)
            self.average_score = np.mean(self.total_score[-10:])
            self.learn()

    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.w)  # simple linear policy
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions

        self.params = "Noise scale: {}".format(self.noise_scale)  # Update the params printout with the latest noise scale
