'''
Testing various TD methods with Tile coding in a continuous state space environment (MountainCar)
Training stops upon getting a return >-100 - the resulting policy doesnt solve the problem everytime
- SARSA,Q-Learning & Expected SARSA
- Corresponding versions with average reward instead of discount
'''

import numpy as np
import gym
import time

from tilecoding import TileCoding

env = gym.make('MountainCar-v0')
nA=env.action_space.n

episodes=1000
step_size=0.5/8
epsilon=0
discount=1

def Q(w,a,x) :
    return np.dot(w[a],x)

def epsilon_greedy_action(w,x,epsilon) :
    A=np.array([epsilon/nA for a in range(nA)])
    best= np.argmax([Q(w,a,x) for a in range(nA)])
    A[best]+=1-epsilon
    return np.random.choice(nA,p=A)

def epsilon_greedy_policy(w,x,epsilon) :
    A=np.array([epsilon/nA for a in range(nA)])
    best= np.argmax([Q(w,a,x) for a in range(nA)])
    A[best]+=1-epsilon
    return A


def sarsa_update(w,a,x,anew,xnew,reward,discount) :
    return (reward+discount*Q(w,anew,xnew)-Q(w,a,x))

def qlearning_update(w,a,x,xnew,reward,discount) :
    return reward+discount*np.max([Q(w,act,xnew) for act in range(nA)])-Q(w,a,x)

def exp_sarsa_update(w,a,x,xnew,reward,discount) :
    return reward+discount*np.dot(epsilon_greedy_policy(w,x,epsilon),np.array([Q(w,act,xnew) for act in range(nA)]).T)-Q(w,a,x)


def sarsa_avg_update(w,a,x,anew,xnew,reward,avgR) :
    return reward-avgR+Q(w,anew,xnew)-Q(w,a,x)

def qlearning_avg_update(w,a,x,xnew,reward,avgR) :
    return reward-avgR+np.max([Q(w,act,xnew) for act in range(nA)])-Q(w,a,x)

def exp_sarsa_avg_update(w,a,x,xnew,reward,avgR) :
    return reward-avgR+np.dot(epsilon_greedy_policy(w,x,epsilon),np.array([Q(w,act,xnew) for act in range(nA)]).T)-Q(w,a,x)

def test_policy (coding,w,env,render=False) :
    A_n=env.action_space.n
    init_state=env.reset()
    if render : env.render()
    observation=env.step(epsilon_greedy_action(w,coding.encode_state(init_state),0))
    ret=0
    for i in range(100) :
        if render : env.render()
        observation=env.step(epsilon_greedy_action(w,coding.encode_state(observation[0]),0))
        ret+=observation[1]
        if observation[2]==True :
            return ret
    return ret

def TD_Tiles(coding,update='qlearning',avg=False) :
    w=np.zeros([nA,8*8*8])
    #w=np.loadtxt('mountaincar_sarsa')
    avgR=0
    avg_ret = -200

    for i in range(episodes) :
        state=env.reset()
        x=coding.encode_state(state)
        a=epsilon_greedy_action(w,x,epsilon)
        done=False
        ret=0
        while True :
            new_state,reward,done,prob=env.step(a)
            xnew=coding.encode_state(new_state)
            anew=epsilon_greedy_action(w,x,epsilon)
            if done :
                w[a]+=step_size*(reward-Q(w,a,x))*x
                break

            if update=='qlearning' : 
                if avg==False : upd=qlearning_update(w,a,x,xnew,reward,discount)
                else          : upd=qlearning_avg_update(w,a,x,xnew,reward,avgR)
            elif update=='sarsa'   : 
                if avg==False : upd=sarsa_update(w,a,x,anew,xnew,reward,discount)
                else          : upd=sarsa_avg_update(w,a,x,anew,xnew,reward,avgR)
            elif update=='exp'     : 
                if avg==False : upd=exp_sarsa_update(w,a,x,xnew,reward,discount)
                else          : upd=exp_sarsa_avg_update(w,a,x,xnew,reward,avgR)
            avgR+=0.2*upd
            w[a]+=step_size*upd*x            
            ret+=reward
            x=xnew
            a=anew
            if i%1000==999 :
                env.render()
        avg_ret = 0.95*avg_ret + 0.05*ret
        if avg_ret>-100 :
            print("\nFinished ! Early exit at episode",i,"with return ",ret)
            print("in {} seconds".format(time.time()-st))
            test_policy(coding,w,env,render=True)
            break

    return w