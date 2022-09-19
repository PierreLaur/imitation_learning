'''
BC functions for use in another file
'''

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from stable_baselines3 import PPO,DQN,DDPG
import gym

env_name = 'CartPole-v1'
env = gym.make(env_name)

def generate_expert_trajectories(env, model, num_trajectories) :
    print("Generating",num_trajectories, "expert trajectories...")
    states = np.empty((0,env.observation_space.shape[0]))
    actions = np.array([])
    for _ in tqdm(range(num_trajectories)) :
        state = env.reset()
        done = False
        while not done :
            action, _ = model.predict([state])
            new_state, _, done, _ = env.step(action[0])
            states = np.vstack((states,state))
            actions = np.append(actions,action[0])
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

def train_expert(env, env_name, num_expert_training_steps, method='PPO') :
    try :
        if method == 'PPO':
            model = PPO.load('policies/PPO_{}_{}'.format(env_name,num_expert_training_steps))
        elif method == 'DQN' :
            model = DQN.load('policies/DQN_{}_{}'.format(env_name,num_expert_training_steps))
        elif method == 'DDPG' :
            model = DDPG.load('policies/DDPG_{}_{}'.format(env_name,num_expert_training_steps))

    except FileNotFoundError :
        print("No expert policy found, training one for ",num_expert_training_steps,"steps")
        if method == 'PPO':
            # train a PPO model with stable-baselines3
            model = PPO("MlpPolicy", env, verbose = 1)
        elif method == 'DQN' :
            # train a DQN model with stable-baselines3
            model = DQN("MlpPolicy", env, verbose = 1,exploration_initial_eps=1,exploration_fraction=0.7)
        elif method == 'DDPG' :
            # train a DDPG model with stable-baselines3
            model = DDPG("MlpPolicy", env, verbose = 1)

        model.learn(total_timesteps=num_expert_training_steps)
        model.save('policies/{}_{}_{}'.format(method,env_name,num_expert_training_steps))
    return model

def fit_to_expert(model,states,actions) :
    model.fit(states,actions)

