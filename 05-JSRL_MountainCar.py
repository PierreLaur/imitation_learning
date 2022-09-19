'''
Trying out the JSRL method to accelerate RL on MountainCar-v0
'''

from collections import deque
import numpy as np
import tensorflow as tf
import os
import gym
from TD_Tiles import TD_Tiles, epsilon_greedy_action
from tilecoding import TileCoding

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def generate_expert_trajectories(w,n_trajs) :
    n_trajs = 100
    print(f'Generating {n_trajs} expert trajectories')

    env = gym.make('MountainCar-v0')
    trajs = []
    n_episodes = 10
    for episode in range(n_episodes) :
        done = False
        state = env.reset()
        score = 0
        traj=[]
        while not done :
            action = epsilon_greedy_action(w, T.encode_state(state), 0)
            new_state,reward,done,info = env.step(action)
            traj.append([state, action, reward, new_state])
            score += reward
            state = new_state
        trajs.append(traj)
        print(f'    Episode {episode} score {score}')
    return trajs

def expert_action(expert,state) :
    T = TileCoding(8,8,8,[-1.2, 0.6],[-0.07,0.07])
    x = T.encode_state(state)
    action = epsilon_greedy_action(expert,x,0)
    return action

def load_expert() :
    env = gym.make('MountainCar-v0')
    try :
        expert = np.loadtxt('policies/Tiles_w')
        print('     Found existing expert')
    except FileNotFoundError :
        T = TileCoding(8,8,8,[-1.2, 0.6],[-0.07,0.07])
        print(' No existing expert found, training one')
        expert = TD_Tiles(T,'qlearning')
        np.savetxt('policies/Tiles_w', w)
    return expert

def test_agent(env, agent, n_episodes, expert=None, guide_steps=0, verbose=1) :
    average_score = 0
    for episode in range(n_episodes) :
        done = False
        state = env.reset()
        score = 0
        step=0
        while not done :
            if step < guide_steps :
                action = expert_action(expert, state)
            else :
                action = agent.select_action(state, greedy=True)
            new_state,reward,done,info = env.step(action)
            score += reward
            state = new_state
            step += 1

        average_score += score
        #print(f'Episode {episode} score {score}')
    total_avg = round(average_score/n_episodes)
    if verbose :
        print(f'    Average evaluation score : {total_avg}')
    return total_avg

def train_agent(env,agent,n_episodes,jsrl=False, store_guide_data=True) :
    if jsrl :
        print('Preparing the expert')
        expert = load_expert()
        expert_score = test_agent(env,None,100,expert,200)
        print(f'\n Training DQN agent with JSRL for max. {n_episodes} episodes...\n')
        h = 200

    else :
        print(f'\n Training DQN agent without JSRL for max. {n_episodes} episodes...\n')
        expert = None
        h = 0

    total_env_steps = 0 # counts the number of steps we have to train
    total_learning_steps = 0
    h_decrease = 5
    for episode in range(n_episodes) :
        done = False
        state = env.reset()
        score = 0
        step = 0
        guide = 0
        exploration = 0
        while not done :
            if step < h :
                action = expert_action(expert,state)
                guide += 1
                new_state,reward,done,info = env.step(action)
                if store_guide_data :
                    agent.store_experience(state,action,reward,new_state,done)

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
        

        if h <= 0 and test_result >= -120 :
            break
        elif jsrl :
            if test_result >= -120 :
                print('         Decreasing guide steps')
                h -= h_decrease


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

print(' -   -   -   Using JSRL to solve MountainCar-v0 with a DQN   -   -   -')

jsrl = int(input('Ready to train. Use JSRL ? (yes : 1, no : 0)\n'))
if jsrl :
    store_guide_data = int(input('Store guide data ? (yes : 1, no : 0)\n'))
n_episodes = 1000
env = gym.make('MountainCar-v0')
n_actions = env.action_space.n
agent = DQNAgent(n_actions)

train_agent(env,agent,n_episodes,jsrl, store_guide_data)

print('\nTraining complete. Testing the agent...')
env = gym.make('MountainCar-v0')
n_episodes = 10
test_agent(env,agent,n_episodes)
