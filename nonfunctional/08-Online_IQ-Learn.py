'''
Implementing IQ-Learn in the Offline setting, starting from a SAC agent
'''

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import gym
from collections import deque
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Qnet (tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = tf.keras.layers.Dense(8, activation='relu')
        self.l2 = tf.keras.layers.Dense(8, activation='relu')
        self.out = tf.keras.layers.Dense(1)
        self.compile(optimizer=tf.keras.optimizers.Adam())

    def call(self, states, actions):
        input = tf.concat([states, actions], axis=1)
        x = self.l1(input)
        x = self.l2(x)
        x = self.out(x)
        return x


class PolicyNet (tf.keras.Model):
    def __init__(self, env) -> None:
        super().__init__()
        self.action_scale = env.action_space.high
        self.l1 = tf.keras.layers.Dense(256, activation='relu')
        self.l2 = tf.keras.layers.Dense(256, activation='relu')
        self.mu = tf.keras.layers.Dense(env.action_space.shape[0])
        self.log_sigma = tf.keras.layers.Dense(env.action_space.shape[0])
        self.compile(optimizer=tf.keras.optimizers.Adam())

    def call(self, input):
        x = self.l1(input)
        x = self.l2(x)
        mu = self.mu(x)

        log_sigma = self.log_sigma(x)
        sigma = tf.exp(log_sigma)

        dist = tfp.distributions.Normal(mu, sigma)
        actions = dist.sample()
        tanh_actions = tf.tanh(actions)
        log_mu = dist.log_prob(actions)
        log_probs = log_mu - \
            tf.reduce_sum(tf.math.log(1-tanh_actions**2+1e-6),
                          axis=1, keepdims=True)

        return actions * self.action_scale, log_probs


class SACAgent:
    def __init__(self, env, discount=0.99, tau=0.01, alpha=1, batch_size=1_000_000, capacity=1_000_000, iq_learn=False) -> None:
        self.n_actions = env.action_space.shape[0]
        self.discount = discount
        self.tau = tau
        self.alpha = alpha


        self.policy = PolicyNet(env)
        self.Q = Qnet()
        self.Q_target = Qnet()
        self.Q_target.set_weights(self.Q.get_weights())

        self.batch_size = batch_size
        replay_buffer_capacity = capacity
        self.states = deque([], maxlen=replay_buffer_capacity)
        self.actions = deque([], maxlen=replay_buffer_capacity)
        self.rewards = deque([], maxlen=replay_buffer_capacity)
        self.new_states = deque([], maxlen=replay_buffer_capacity)
        self.dones = deque([], maxlen=replay_buffer_capacity)

        self.iq_learn = iq_learn
        if iq_learn :
            self.expert_states = []
            self.expert_actions = []
            self.expert_new_states = []
            self.expert_dones = []

    def load_expert_data(self,states,actions,new_states,dones) :
        self.expert_states = states
        self.expert_actions = actions
        self.expert_new_states = new_states
        self.expert_dones = dones

    def sample_expert(self) :
        rand = np.random.choice(np.arange(len(self.expert_states)))
        return self.expert_states[rand], self.expert_actions[rand], self.expert_new_states[rand], self.expert_dones[rand]

    def sample_buffer(self) :
        rand = np.random.choice(np.arange(len(self.states)))
        return self.states[rand], self.actions[rand], self.new_states[rand], self.dones[rand]

    def update_target(self):
        new_weights = []
        for target, main in zip(self.Q_target.get_weights(), self.Q.get_weights()):
            new_weights.append(
                self.tau * main + (1-self.tau) * target
            )
        self.Q_target.set_weights(new_weights)

    def select_action(self, state):
        state = tf.convert_to_tensor([state])
        action, _ = self.policy(state)
        return action[0]

    def store_experience(self, state, action, reward, new_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)
        self.dones.append(done)

    def generate_batch(self):
        batch_indices = np.random.choice(
            np.arange(len(self.states)), size=min(self.batch_size, len(self.states))
        )
        states = np.array(self.states)[batch_indices]
        actions = np.array(self.actions)[batch_indices]
        rewards = np.array(self.rewards)[batch_indices]
        new_states = np.array(self.new_states)[batch_indices]
        dones = np.array(self.dones)[batch_indices]
        return states, actions, rewards, new_states, dones

    def learn(self):

        states, actions, rewards, new_states, dones = self.generate_batch()

        if self.iq_learn :
            with tf.GradientTape() as tape :
                new_actions, new_log_probs = self.policy(self.expert_new_states)
                v = self.Q_target(self.expert_new_states, new_actions) - self.alpha * new_log_probs
                print(f'{v.shape=}\n{new_log_probs.shape=}\n{self.expert_new_states.shape=}')
                current_actions, current_log_probs = self.policy(self.expert_new_states)
                current_q = tf.squeeze(self.Q(self.expert_states, current_actions), 1)
                iq_loss = tf.reduce_mean(current_q - (1-dones) * self.discount * tf.squeeze(v,1))
                # sample half from expert data, half from replay buffer
                if np.random.uniform() < 0.5 :
                    sample_state,sample_action,sample_new_state,sample_done = self.sample_buffer()
                else :
                    sample_state,sample_action,sample_new_state,sample_done = self.sample_expert()
                sample_state = np.reshape(sample_state,(1,-1))
                sample_new_state = np.reshape(sample_new_state,(1,-1))

                sample_current_action, sample_current_log_probs = self.policy(sample_state)
                v = self.Q_target(sample_state,sample_current_action) - self.alpha * sample_current_log_probs
                sample_current_new_action, sample_current_new_log_probs = self.policy(sample_new_state)
                new_v = self.Q_target(sample_new_state,sample_current_new_action) - self.alpha * sample_current_new_log_probs
                iq_loss -= tf.reduce_mean((1-self.discount) * (tf.squeeze(v,1) - (1-sample_done) * self.discount * tf.squeeze(new_v,1)))

            q_grads = tape.gradient(-iq_loss, self.Q.trainable_variables)
            self.Q.optimizer.apply_gradients(
                zip(q_grads, self.Q.trainable_variables))
        else :
            # Update Q function
            with tf.GradientTape() as tape:
                q = tf.squeeze(self.Q(states, actions), 1)
                new_actions, log_probs = self.policy(new_states)
                v = self.Q_target(new_states, new_actions) - self.alpha * log_probs
                better_q = rewards + (1-dones) * self.discount * tf.squeeze(v, 1)
                q_loss = 0.5*tf.keras.losses.MSE(q, better_q)
            q_grads = tape.gradient(q_loss, self.Q.trainable_variables)
            self.Q.optimizer.apply_gradients(
                zip(q_grads, self.Q.trainable_variables))


        # Update policy
        with tf.GradientTape() as tape:
            new_actions, log_probs = self.policy(states)
            q = self.Q(states, new_actions)
            pi_loss = tf.reduce_mean(self.alpha * log_probs - q)
        pi_grads = tape.gradient(pi_loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(
            zip(pi_grads, self.policy.trainable_variables))

        self.update_target()


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
            action = agent.select_action(state)
            new_state,reward,done,info = env.step(action)
            states.append(state)
            actions.append(action)
            score += reward
            state = new_state
            
        if score == 1000 :
            demo_states.append(states)
            demo_actions.append(actions)
            print("Validating 1 trajectory...")
        else :
            print(f"Obtained score {score}, retrying")
    for i in range(n_trajectories) :
        np.savetxt(f'policies/InvertedPendulum_expert_states_{i}', demo_states[i])
        np.savetxt(f'policies/InvertedPendulum_expert_actions_{i}', demo_actions[i])

def train_expert(env,agent) :
    n_episodes = 10000
    average_score = 15
    print('Training a SAC Agent...')
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
        if score == 1000 :
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
            action = agent.select_action(state)
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
            demo_states.append(np.loadtxt(f'policies/InvertedPendulum_expert_states_{i}'))
            demo_actions.append(np.loadtxt(f'policies/InvertedPendulum_expert_actions_{i}'))
    except FileNotFoundError :
        print('Expert demonstrations not found, generating...')
        expert = SACAgent(env)
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
env = gym.make('InvertedPendulum-v2')

print(' -   -   -   Testing Online IQ-Learn on InvertedPendulum-v2 -   -   -\n')
num_demos = int(input("Number of expert demonstrations ?"))

print(f'Loading {num_demos} expert demonstrations...')
demo_states, demo_actions, demo_new_states, demo_dones = load_demos(env,num_demos)
states = np.vstack([demo_states[i] for i in range(num_demos)])
actions = np.concatenate([demo_actions[i] for i in range(num_demos)])
new_states = np.vstack([demo_new_states[i] for i in range(num_demos)])
dones = np.concatenate([demo_dones[i] for i in range(num_demos)])


n_episodes = 500
average_score = 0
agent = SACAgent(env,iq_learn=True)
agent.load_expert_data(states,actions,new_states,dones)
for episode in range(n_episodes):
    done = False
    state = env.reset()
    score = 0
    while not done:
        action = agent.select_action(state)
        new_state, reward, done, info = env.step(action)
        agent.store_experience(state, action, reward, new_state, done)
        # env.render()
        score += reward
        state = new_state

        agent.learn()

    average_score = 0.95*average_score + 0.05*score
    print(
        f'Episode {episode} score {score}      average score {round(average_score/(1-0.95**(episode+1)))}')
