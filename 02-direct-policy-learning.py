'''
Direct Policy Learning (with an interactive demonstrator)
'''

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from stable_baselines3 import PPO

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gym
env = gym.make('CartPole-v1')

def generate_trajectories(env, model, num_trajectories, hide_progress=False) :
    states = np.empty((0,env.observation_space.shape[0]))
    actions = np.array([])
    for _ in tqdm(range(num_trajectories), disable=hide_progress) :
        state = env.reset()
        done = False
        while not done :
            nn_input = tf.convert_to_tensor([state])
            action = model.predict(nn_input)[0]
            action = 0 if action<0.5 else 1
            new_state, _, done, _ = env.step(action)
            states = np.vstack((states,state))
            actions = np.append(actions,action)
            state = new_state
    return states, actions

class BC_net (tf.keras.Model) :
    def __init__(self) -> None:
        super().__init__()
        self.l1 = tf.keras.layers.Dense(4, activation='relu')
        self.l2 = tf.keras.layers.Dense(4, activation='relu')
        self.l3 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.compile(optimizer = "Adam", loss = "BCE")

    def call(self, input) :
        x = self.l1(input)
        x = self.l2(x)
        x = self.l3(x)
        return x

    def predict(self,input) :
        return super().predict(input,verbose=0)

def test_policy(env,policy,num_episodes) :
    scores = []
    for i in range(num_episodes) :
        done = False
        state = env.reset()
        score = 0
        while not done :
            nn_input = tf.convert_to_tensor([state])
            action = policy.predict(nn_input)[0][0]
            action = 0 if action<0.5 else 1
            new_state, reward, done, info = env.step(action)
            env.render()
            score += reward
            state = new_state
        print("Ep",i,"score",score)
        scores.append(score)

    print("\n Average score :",np.mean(scores))

def train_expert(env, env_name, num_expert_training_steps) :
    try :
        model = PPO.load('policies/PPO_{}_{}'.format(env_name,num_expert_training_steps))
    except FileNotFoundError :
    # train a PPO model with stable-baselines3
        print("No expert policy found, training one for ",num_expert_training_steps,"steps")
        model = PPO("MlpPolicy", env, verbose = 1)
        model.learn(total_timesteps=num_expert_training_steps)
        model.save('policies/PPO_{}_{}'.format(env_name,num_expert_training_steps))
    return model

def aggregate(states,actions,new_states,new_actions) :
    states = np.vstack((states,new_states))
    actions = np.append(actions,new_actions)
    return states,actions

# HPs
n_expert_training_steps = 200000
n_expert_trajectories = 10
n_iterations = 50
n_rollouts = 1

apprentice = BC_net()
print("- - Loading expert policy")
expert = train_expert(env,'CartPole-v1',n_expert_training_steps)

# Iteration 0 - basically BC
print("- - Initializing (BC on expert trajectories)")
states, actions = generate_trajectories(env,expert,n_expert_trajectories)
apprentice.fit(states,actions,verbose=0)

# Sequential Learning loop
print("- - Training the apprentice")
for i in tqdm(range(n_iterations)) :

    # Generate trajectories by rolling out the current apprentice policy
    new_states, new_actions = generate_trajectories(env,apprentice,n_rollouts,hide_progress=True)

    # Query the expert
    expert_actions = expert.predict(new_states)

    # Aggregate the expert data (DAgger technique)
    states, actions = aggregate(states,actions,new_states,new_actions)
    
    # Train the apprentice
    apprentice.fit(states,actions,verbose=0)

print("- - Training finished - testing the resulting apprentice policy")
test_policy(env,apprentice,5)

