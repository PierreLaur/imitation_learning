'''
Implementing IQ-Learn in the offline setting
Doesn't work as of now (07/09)
'''

import numpy as np
import tensorflow as tf
import gym
import os
from collections import deque
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Qnetwork (tf.keras.Model) :
   def __init__(self,n_actions,lr,layers_dim) -> None:
      super().__init__()
      self.l1 = tf.keras.layers.Dense(layers_dim,activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
      self.l2 = tf.keras.layers.Dense(layers_dim,activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
      self.out = tf.keras.layers.Dense(n_actions,activation='linear')
      self.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(lr))

   def call(self,input) :
      x = self.l1(input)
      x = self.l2(x)
      x = self.out(x)
      return x

class Agent:
    def __init__(self, env,batch_size,discount,lr,layers_dim) -> None:
        self.n_actions = env.action_space.n
        self.epsilon = 0.3
        self.discount = discount
        self.batch_size = batch_size

        self.Q_function = Qnetwork(self.n_actions,lr,layers_dim)
        
        replay_buffer_capacity = 1000000
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

    def store_experience(self, states, actions, rewards, new_states, dones):
        self.states=states
        self.actions=actions
        self.rewards=rewards
        self.new_states=new_states
        self.dones=dones

    def generate_batches(self) :
        states = []
        actions = []
        new_states = []
        dones = []
        for i in range(len(self.states)//self.batch_size) :
            batch_indices = np.random.choice(np.arange(i*self.batch_size,(1+i)*self.batch_size),size=self.batch_size)
            states.append(np.array(self.states)[batch_indices])
            actions.append(np.array(self.actions,dtype=np.int64)[batch_indices])
            new_states.append(np.array(self.new_states)[batch_indices])
            dones.append(np.array(self.dones)[batch_indices])
        return states,actions,new_states,dones

    def learn(self):
        if len(self.states) < self.batch_size : 
            return

        states,actions,new_states,dones = self.generate_batches()

        for i,(states,actions,new_states,dones) in enumerate(zip(states,actions,new_states,dones)) :
            with tf.GradientTape() as tape :
                # WITH EXPERT STATES & ACTIONS
                q_s = self.Q_function(states)
                new_q_s = self.Q_function(new_states)
                v_s = tf.reduce_logsumexp(q_s, axis = 1)
                new_v_s = tf.reduce_logsumexp(new_q_s, axis = 1)
                y = self.discount * new_v_s * (1-dones)
                new_q_s_a = new_q_s.numpy()[np.arange(self.batch_size),actions]
                expected_iq_reward = tf.reduce_mean(new_q_s_a - y)             # iq_reward can be transformed to use different divergence functions
                expected_initial_v = tf.reduce_mean(v_s - y)                    # this can be sampled in other ways
                loss = - (expected_iq_reward - expected_initial_v) 
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

def test_agent(env, agent, n_episodes, set_seed=False,verbose=1) :
    average_score = 0
    for episode in range(n_episodes) :
        done = False
        if set_seed :
            env.seed(episode)
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

def load_demos(env,num_demos) :
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
            demo_actions.append(np.array(np.loadtxt(f'policies/CartPole_expert_actions_{i}'),dtype=np.int64))
    demo_new_states = []
    demo_dones = []
    for i in range(num_demos) :
        demo_new_states.append(np.roll(demo_states[i],shift=-1, axis=0))
        z = np.zeros(shape=(len(demo_states[i])))
        z[-1] = 1
        demo_dones.append(z)
    return demo_states,demo_actions,demo_new_states,demo_dones

# np.random.seed(3)
# tf.random.set_seed(3)
env = gym.make('CartPole-v1')

print(' -   -   -   Testing Offline IQ-Learn on CartPole-v1 -   -   -\n')
num_demos = int(input("Number of expert demonstrations ?"))
print(f'Loading {num_demos} expert demonstrations...')
demo_states, demo_actions, demo_new_states, demo_dones = load_demos(env,num_demos)

states = np.vstack([demo_states[i] for i in range(num_demos)])
new_states = np.vstack([demo_new_states[i] for i in range(num_demos)])
actions = np.concatenate([demo_actions[i] for i in range(num_demos)])
dones = np.concatenate([demo_dones[i] for i in range(num_demos)])

# Hyperparameters
batch_size = 256
discount = 0.99
lr = 0.001
layers_dim = 32
n_epochs = 100000

agent = Agent(env,batch_size,discount,lr,layers_dim)

agent.store_experience(states,actions,None,new_states,dones)
avg_score = 9
for ep in range(n_epochs) :
    agent.learn()

    avg_score = 0.97*avg_score + 0.03*test_agent(env,agent,5,set_seed=True, verbose=0)
    print(f'Epoch {ep} average score {round(avg_score)}')
    
