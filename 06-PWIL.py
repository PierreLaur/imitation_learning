'''
Implementing Primal Wasserstein Imitation Learning
'''

import numpy as np
import tensorflow as tf
import gym
import os
from collections import deque
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Qnet (tf.keras.Model) :
   def __init__(self,n_actions) -> None:
      super().__init__()
      self.l1 = tf.keras.layers.Dense(24,activation='relu')
      self.l2 = tf.keras.layers.Dense(24,activation='relu')
      self.out = tf.keras.layers.Dense(n_actions,activation='linear')
      self.compile(optimizer=tf.keras.optimizers.Adam())

   def call(self,input) :
      x = self.l1(input)
      x = self.l2(x)
      x = self.out(x)
      return x

class Agent:
    def __init__(self, env) -> None:
        self.n_actions = env.action_space.n
        self.epsilon = 0.3
        self.discount = 0.99
        self.batch_size = 512

        self.Q_function = Qnet(self.n_actions)
        
        replay_buffer_capacity = 10000
        self.states = deque([],maxlen=replay_buffer_capacity)
        self.actions = deque([],maxlen=replay_buffer_capacity)
        self.rewards = deque([],maxlen=replay_buffer_capacity)
        self.new_states = deque([],maxlen=replay_buffer_capacity)
        self.dones = deque([],maxlen=replay_buffer_capacity)

    def select_action(self, state, greedy=False):
        if not greedy and np.random.uniform(0,1) < self.epsilon :
            action = np.random.choice(self.n_actions)
            return action
        else :
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

    def generate_batch(self) :
        def normalized(input) :
            return (input - np.mean(self.rewards)) / np.std(self.rewards)
        batch_indices = np.random.choice(np.arange(len(self.states)),size=self.batch_size)
        states = np.array(self.states)[batch_indices]
        actions = np.array(self.actions)[batch_indices]
        rewards = np.array(self.rewards)[batch_indices]
        # normalized_rewards = normalized(rewards)
        new_states = np.array(self.new_states)[batch_indices]
        dones = np.array(self.dones)[batch_indices]
        return states,actions,rewards,new_states,dones

    def learn(self):
        if len(self.states) < self.batch_size :
            return

        states,actions,rewards,new_states,dones = self.generate_batch()
        
        with tf.GradientTape() as tape :
            values = self.Q_function(states)
            updates = rewards + self.discount * tf.reduce_max(self.Q_function(new_states),axis=1) * (1-dones)
            targets = values.numpy()
            targets[np.arange(self.batch_size),actions] = updates
            targets = tf.convert_to_tensor(targets)
            loss = tf.keras.losses.MSE(values,targets)
        grads = tape.gradient(loss, self.Q_function.trainable_variables)
        self.Q_function.optimizer.apply_gradients(zip(grads,self.Q_function.trainable_variables))

def generate_expert_demonstrations(env,agent,n_trajectories) :
    demo_states = []
    demo_actions = []
    print(f"Generating {n_trajectories} trajectories...")
    while len(demo_states) < n_trajectories :
        done = False
        state = env.reset()
        states = []
        actions = []
        score = 0
        while not done :
            action = agent.select_action(state,greedy=True)
            new_state,reward,done,info = env.step(action)
            states.append(state)
            actions.append(action)
            score += reward
            state = new_state
            
        if score == 500 :
            demo_states.append(states)
            demo_actions.append(actions)
            print("Validating 1 trajectory...")
        else :
            print(f"Obtained score {score}, retrying")
    for i in range(n_trajectories) :
        np.savetxt(f'policies/CartPole_expert_states_{i}', demo_states[i])
        np.savetxt(f'policies/CartPole_expert_actions_{i}', demo_actions[i])

def train_expert(env,agent) :
    n_episodes = 10000
    average_score = 15
    print('Training a DQN Agent...')
    st = time.process_time()
    for episode in range(n_episodes) :
        done = False
        state = env.reset()
        score = 0
        while not done :
            action = agent.select_action(state)
            new_state,reward,done,info = env.step(action)
            agent.store_experience(state,action,reward,new_state,done)
            score += reward
            state = new_state
            agent.learn()

        average_score = 0.90*average_score + 0.10*score
        print(f'    Episode {episode} score {score}      average score {round(average_score)}')
        if score == 500 :
            print(f'Time taken to train with RL : {time.process_time()-st}')
            break

def test_agent(env, agent, n_episodes, verbose=1) :
    average_score = 0
    for episode in range(n_episodes) :
        done = False
        state = env.reset()
        score = 0
        step=0
        while not done :
            action = agent.select_action(state, greedy=True)
            new_state,reward,done,info = env.step(action)
            score += reward
            state = new_state
            step+=1

        average_score += score
        if verbose == 2 :
            print(f'Episode {episode} score {score}')
    total_avg = round(average_score/n_episodes)
    if verbose>=1 :
        print(f'    Average evaluation score : {total_avg}')
    return total_avg

env = gym.make('CartPole-v1')
agent = Agent(env)

print(f"    -   -   -   Testing Primal Wasserstein Imitation Learning on CartPole-v1    -   -   -\n")

num_demos = int(input("Number of expert demonstrations ?"))
print(f'Loading {num_demos} expert demonstrations...')
demo_states = []
demo_actions = []
try :
    for i in range(num_demos) :
        demo_states.append(np.loadtxt(f'policies/CartPole_expert_states_{i}'))
        demo_actions.append(np.loadtxt(f'policies/CartPole_expert_actions_{i}'))
except FileNotFoundError :
    print('Expert demonstrations not found, generating...')
    expert = Agent(env)
    train_expert(env,expert)                                            # To train an agent enough to generate expert demonstrations
    test_agent(env,expert,10, verbose=2)
    generate_expert_demonstrations(env,expert,n_trajectories=num_demos)       # To generate expert demos using the agent
    demo_states = []
    demo_actions = []
    for i in range(num_demos) :
        demo_states.append(np.loadtxt(f'policies/CartPole_expert_states_{i}'))
        demo_actions.append(np.loadtxt(f'policies/CartPole_expert_actions_{i}'))

# The algorithm

n_episodes = 10000
average_score = 10
horizon = 500
alpha = 0.001
beta = 1
sigma = alpha * horizon / np.sqrt(env.action_space.n + env.observation_space.shape[0])
st = time.process_time()
print('Training agent with PWIL...')
for episode in range(n_episodes) :
    done = False
    state = env.reset()
    score = 0
    chosen_demo = np.random.choice(num_demos)
    chosen_demo_actions = demo_actions[chosen_demo]
    chosen_demo_states = demo_states[chosen_demo]
    D = len(chosen_demo_actions)
    weights = [1/D for i in range(D)]     # 1/Total number of expert state-action pairs (1/D in the paper)

    while not done :                                            # handling horizon T ?
        action = agent.select_action(state)                     
        new_state, reward, done, prob = env.step(action)        
        score += reward
        
        # The weights should sum to 1 - note that it isnt the case due to variable episode length
        weight_pi = 1/horizon
        cost = 0
        # while there is still dirt
        while weight_pi > 0 :
            # Finding the greedy coupling (the closest hole)
            diff = np.sum(np.abs(chosen_demo_states[:,0] - state[0])+np.abs(chosen_demo_states[:,2] - state[2]))
            closest = np.argmin(diff)

            # If there is more 'dirt mass' (weight_pi) than space in the hole (expert weight)
            if weight_pi >= weights[closest] :
                cost += weights[closest] * np.sum(np.abs(chosen_demo_states[:,0] - state[0])+np.abs(chosen_demo_states[:,2] - state[2]))
                weight_pi -= weights[closest]

                chosen_demo_states = np.delete(chosen_demo_states,closest,0)
                chosen_demo_actions = np.delete(chosen_demo_actions,closest)
                weights = np.delete(weights,closest)
            else :
                cost += weight_pi * np.sum(np.abs(chosen_demo_states[:,0] - state[0])+np.abs(chosen_demo_states[:,2] - state[2]))
                weights[closest] -= weight_pi
                weight_pi = 0

        reward = beta * np.exp(-sigma*cost)
        
        agent.store_experience(state,action,reward,new_state,done)
        agent.learn()
        state = new_state

    average_score = 0.90*average_score + 0.10*score
    print(f'    Episode {episode} score {score}      average score {round(average_score)}')
    if episode%10 == 0 and test_agent(env,agent,5,verbose=0) >= 450 :
        print(f'Time taken to train with PWIL : {time.process_time()-st}')
        break


print(f'Training complete, testing the agent')
test_agent(env,agent,10,verbose = 2)
