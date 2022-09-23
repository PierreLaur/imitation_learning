'''
Trying out the JSRL method to accelerate a DQN agent on CartPole-v1
It is not needed (the problem is already simple), but it works
from Uchendu et al. 2022
'''

from collections import deque
import numpy as np
import tensorflow as tf
import os
import gym
import behavioral_cloning
from stable_baselines3 import PPO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_agent(env, agent, n_episodes, expert=None, guide_steps=0, verbose=1) :
    average_score = 0
    for episode in range(n_episodes) :
        done = False
        state = env.reset()
        score = 0
        step=0
        while not done :
            if step < guide_steps :
                action, _ = expert.predict(state)
            else :
                action = agent.select_action(state, greedy=True)
            new_state,reward,done,info = env.step(action)
            score += reward
            state = new_state
            step+=1

        average_score += score
        #print(f'Episode {episode} score {score}')
    total_avg = round(average_score/n_episodes)
    if verbose :
        print(f'    Average evaluation score : {total_avg}')
    return total_avg

class ValueNet(tf.keras.Model):
    def __init__(self, n_actions) -> None:
        super().__init__()
        self.l1 = tf.keras.layers.Dense(24, activation='relu')
        self.l2 = tf.keras.layers.Dense(24, activation='relu')
        self.out = tf.keras.layers.Dense(n_actions, activation='linear')
        self.compile(loss='MSE', optimizer="Adam")

    def call(self, input):
        x = self.l1(input)
        x = self.l2(x)
        x = self.out(x)
        return x

class DQNAgent:
    def __init__(self, n_actions, double=False) -> None:
        self.n_actions = n_actions
        self.Q_function = ValueNet(n_actions)
        self.epsilon = 0.3
        self.batch_size = 512
        self.discount = 0.99
        self.min_capacity = 20000
        self.max_capacity = 20000

        self.states = deque([], maxlen=self.max_capacity)
        self.actions = deque([], maxlen=self.max_capacity)
        self.rewards = deque([], maxlen=self.max_capacity)
        self.new_states = deque([], maxlen=self.max_capacity)
        self.dones = deque([], maxlen=self.max_capacity)

        if double:
            self.double = True
            self.Q_aux = ValueNet(n_actions)
        else:
            self.double = False

    def select_action(self, state, greedy=False):
        if not greedy and np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_actions)
            return action
        else:
            input = tf.convert_to_tensor([state])
            q_values = self.Q_function(input)
            action = tf.argmax(q_values[0]).numpy()
            return action
    
    def buffer_size(self) :
        return len(self.states)

    def store_experience(self, state, action, reward, new_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)
        self.dones.append(done)

    def generate_batch(self):
        batch_indices = np.random.choice(
            np.arange(len(self.states)), size=self.batch_size)
        states = np.array(self.states)[batch_indices]
        actions = np.array(self.actions)[batch_indices]
        rewards = np.array(self.rewards)[batch_indices]
        new_states = np.array(self.new_states)[batch_indices]
        dones = np.array(self.dones)[batch_indices]
        return states, actions, rewards, new_states, dones

    def average_weights(self):
        Q_weights = self.Q_function.get_weights()
        aux_weights = self.Q_aux.get_weights()

        new_aux_weights = []

        for q, aux in zip(Q_weights, aux_weights):
            new_aux_weights.append(0.01*q + 0.99*aux)

        self.Q_aux.set_weights(new_aux_weights)

    def learn(self):
        if len(self.states) < self.batch_size:
            return 0

        states, actions, rewards, new_states, dones = self.generate_batch()

        states = tf.convert_to_tensor(states)
        new_states = tf.convert_to_tensor(new_states)

        q_values = self.Q_function(states).numpy()
        new_q_values = self.Q_function(new_states).numpy()

        if self.double:
            aux_q_values = self.Q_aux(new_states).numpy()
            updates = rewards + self.discount * \
                new_q_values[np.arange(self.batch_size), np.argmax(
                    aux_q_values, axis=1)] * (1-dones)
        else:
            updates = rewards + self.discount * \
                np.max(new_q_values, axis=1) * (1-dones)

        q_values[np.arange(self.batch_size), actions] = updates
        targets = tf.convert_to_tensor(q_values)

        self.Q_function.fit(states, targets, verbose=0)

        if self.double:
            self.average_weights()

        return 1

def train_agent(env,agent,n_episodes,jsrl=False) :
    print('Preparing the expert')
    if jsrl :
        try :
            expert = PPO.load('policies/PPO_CartPole-v1_200000.zip',env)
        except :
            print("Expert not found, training one")
            expert = behavioral_cloning.train_expert(
                        env, 'CartPole-v1', 200000)
        print(f'\n Training DQN agent with JSRL for max. {n_episodes} episodes...\n')
        h = 500

    else :
        print(f'\n Training DQN agent without JSRL for max. {n_episodes} episodes...\n')
        expert = None
        h = 0

    max_obtained = 0 # counts the number of times we obtained the maximum score to detect convergence
    total_env_steps = 0 # counts the number of steps we have to train
    total_learning_steps = 0
    for episode in range(n_episodes) :
        done = False
        state = env.reset()
        score = 0
        step = 0
        guide = 0
        exploration = 0
        while not done :
            if step < h :
                action, _ = expert.predict(state)
                guide += 1
                new_state,reward,done,info = env.step(action)
            else :
                action = agent.select_action(state)
                exploration += 1
                new_state,reward,done,info = env.step(action)
            agent.store_experience(state,action,reward,new_state,done)
            score += reward
            state = new_state
            step += 1
            total_env_steps += 1

            total_learning_steps += agent.learn()

        guide_perc = f'guide used for {guide} steps out of {guide+exploration}' if jsrl else '' # {round(100*guide/(guide+exploration))} % of the time' if jsrl else ''
        print(f'Episode {episode} score {round(score)}       {guide_perc}           {total_env_steps=} {total_learning_steps=}')

        test_result = test_agent(env, agent, 10, expert, h)
        print(f'         Buffer size : {agent.buffer_size()}')

        if h <= 0 :
            if test_result == 500 :
                max_obtained += 1
            else :
                max_obtained = 0
        elif test_result > 490 :
            print('         Decreasing guide steps')
            h -= 5

        # We assume convergence if we obtain the maximum average score 10 times in a row (equivalent to 100 trials)
        if max_obtained >= 10 :
            break

print(' -   -   -   Using JSRL to solve MountainCar-v0 with a DQN   -   -   -')

jsrl = int(input('Ready to train. Use JSRL ? (yes : 1, no : 0)\n'))
n_episodes = 1000
env = gym.make('CartPole-v1')
n_actions = env.action_space.n
agent = DQNAgent(n_actions, double=True)

train_agent(env,agent,n_episodes,jsrl)

print('\nTraining complete. Testing the agent...')
env = gym.make('CartPole-v1')
n_episodes = 100
test_agent(env,agent,n_episodes)
