import numpy as np
import random
from collections import deque
from .actor import Actor
from .critic import Critic
from .noise import OUNoise
from .replay_buffer import ReplayBuffer

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task,
        mu=0, theta=0.15, sigma=0.22,
        buffer_size=1000000, batch_size=128,
        gamma=.99, tau=0.02, learning_rate=0.001):
        """
        Params
        ======
            task: Task (environment) that defines the goal and provides feedback to the agent
            mu (float): Mu in the OU noise algorithm
            theta (float): Theta in the OU noise algorithm
            sigma (float): Sigma in the OU noise algorithm
            buffer_size (int): Maximum size of the buffer in the replay memory
            batch_size (int): Size of each training batch in the replay memory
            gamma (float): Discount factor
            tau (float): Soft update of target parameters
            learning_rate (float): the learning rate for the optimizers in the actor & critic models
        """
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        self.learning_rate = learning_rate

        # Actor (Policy) Model
        # self.actor_lr = learning_rate
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.learning_rate)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.learning_rate)

        # Critic (Value) Model
        # self.critic_lr = learning_rate
        self.critic_local = Critic(self.state_size, self.action_size, self.learning_rate)
        self.critic_target = Critic(self.state_size, self.action_size, self.learning_rate)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = mu  # Default value was 0
        self.exploration_theta = theta  # Default value was 0.15
        self.exploration_sigma = sigma  # Default value was 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = buffer_size  # Default value was 100000
        self.batch_size = batch_size  # Default value was 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = gamma # Default value was 0.99
        self.tau = tau  # Default value was 0.01

        # Score tracker
        self.best_score = -np.inf
        self.average_score = 0.  # This value is really the recent average, based on the last 10 episodes
        self.total_score = []

        self.params = {"Mu":self.exploration_mu, "Theta":self.exploration_theta,
            "Sigma":self.exploration_sigma, "Gamma":self.gamma,
            "Tau":self.tau,"Learning Rate":self.learning_rate}

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.score = 0.
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)
        self.score += reward
        if done:
            self.total_score.append(self.score)
            self.average_score = np.mean(self.total_score[-10:])
            if self.score > self.best_score:
                self.best_score = self.score

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)



class DeepQ():
    """ """
    def __init__(self, task, gamma=0.95,
        epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
        learning_rate=0.001, memory_len=4000, batch_size=128):
        """
        Params
        ======
            task: Task (environment) that defines the goal and provides feedback to the agent
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_min: Minimum exploration
            epsilon_decay: How much to decrease exploration as the agent gets better
            learning_rate: How broadly (high #s) or narrowly (low #s) the agent learns in each iteration
            memory_len: How many past experiences the agent remembers
            batch_size: Size of each training batch in the replay memory
        """
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_len)
        self.batch_size =  batch_size

        self.params = {"Gamma":self.gamma, "Epsilon":self.epsilon,
            "Min Epsilon":self.epsilon_min, "Epsilon Decay":self.epsilon_decay, "Learning Rate":self.learning_rate}

        # Score tracker and learning parameters
        self.best_score = -np.inf
        self.average_score = 0.
        self.total_score = []

        self.model = self._build_model()
        self.target_model = self._build_model()

        self.reset_episode()

    def reset_episode(self):
        state = self.task.reset()
        self.score = 0.
        self.last_state = state
        return state

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        ## Adds exploration
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.rand() <= self.epsilon:
            action = [random.uniform(self.action_low, self.action_high) for i in range(self.action_size)]
            return action

        state = np.reshape(state, [-1, self.state_size])
        action = self.model.predict(state)
        return action[0]

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.remember(self.last_state, action, reward, next_state, done)
        self.score += reward
        if done:
            self.total_score.append(self.score)
            self.average_score = np.mean(self.total_score[-10:])
            if self.score > self.best_score:
                self.best_score = self.score

        # Roll over last state and action
        self.last_state = next_state

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            self.replay(self.batch_size)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            next_state = np.reshape(next_state, [-1, self.state_size])
            state = np.reshape(state, [-1, self.state_size])
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            if not done:
                target[0][action] = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
