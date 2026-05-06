import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def run(episodes,render=False):
    env=gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False,render_mode='human' if render else None)
    q=np.zeros((env.observation_space.n, env.action_space.n))
    learning_rate=0.1
    discount_factor=0.9
    
    epsilon= 1
    epsilon_decay_rate=0.0001
    rng=np.random.default_rng()
    
    reward_per_episode=np.zeros(episodes)
    
    for i in range(episodes):
        state=env.reset()[0]
        terminated=False
        truncated=False
        
        while (not terminated and not truncated):
            action=env.action_space.sample()
            
            new_stage,reward,terminated,truncated,_=env.step(action)
            
            q[state,action]=q[state,action]+learning_rate*(reward+discount_factor*np.max(q[new_stage])-q[state,action])
            
            state=new_stage
        epsilon=max(epsilon-epsilon_decay_rate,0)
    
        if (epsilon==0):
            learning_rate=0.0001
        
        if reward==1:
            reward_per_episode[i]=1

    env.close()
    
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t]=np.sum(reward_per_episode[max(0,t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('reward_per_episode.png')

if __name__ =='__main__':
    run(15000)