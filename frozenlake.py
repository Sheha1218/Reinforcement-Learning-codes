import gymnasium as gym
import numpy as np

def run(episodes):
    env = gym.make('FrozenLake-v1',map_name='8x8',is_slippery=True,render_mode='human')
    
    q =np.zeros(env.observation_space.n, env.action_space.n)
    
    learing_rate=0.9
    discount_factor=0.9
    
    epsilon =1
    epsilon_dacay =0.0001
    rng =np.random.default_rng()
    
    
    
    for i in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
    
        reward_per_episode =np.zeros(episodes)
        while not (terminated or truncated):
            if rng.random()< epsilon:
                action = env.action_space.sample()
                
            else:
                action =np.argmax(q[state,:])
                
            state, reward, terminated, truncated, info = env.step(action)
            
            q[state,action] = q[state,action] + learing_rate * (reward + discount_factor * np.max(q[state,:]- q[state,action]))
        
        epsilon = max(epsilon - epsilon_dacay,0)
        
        if (epsilon==0):
            learing_rate = 0.0001
            
        if reward ==1:
            reward_per_episode[i]=1
    
        
        env.close()
        
        

if __name__ == "__main__":
    run()
