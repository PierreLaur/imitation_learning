# imitation_learning
Minimal implementations of Imitation Learning algorithms.  
Mainly done during my internship at 3IT Sherbrooke - Humanoid and Collaborative Robotics team  
Please feel free to email me if you have any questions.  
Pierre Laur [plaur@insa-toulouse.fr](mailto:plaur@insa-toulouse.fr)  

- Install requirements with ```pip install -r requirements.txt```
- Files that start with a number can be directly launched, the rest are for usage in other files

* Includes implementations of Behavioral Cloning, DAgger direct policy learning, and [Primal Wasserstein Imitation Learning](https://arxiv.org/abs/2006.04678)
* Includes an implementation of [Jump-Start Reinforcement Learning](https://arxiv.org/abs/2204.02372)
	- on CartPole-v1 to try it (it is not needed)
	- on MountainCar-v0 (hard exploration problem)
		* JSRL allows a DQN to solve it
* [rl_experiments submodule](https://github.com/PierreLaur/imitation_learning) includes minimal implementations of various RL algorithms (SAC, PPO, DQN, DDPG...) along with a few other experiments 
