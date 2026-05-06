import gymnasium as gym
import torch
import numpy as np 
import random
import torch.nn as nn
import torch.optim as optm
from collections import deque
import torch.functional as F



class DQN(nn.Module):
    def __init__(self,in_states,hidden,out_actions):
        super().__init__()
        self.fc1=nn.Linear(in_states,hidden)
        self.out =nn.Linear(hidden,out_actions)
        
    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.out(x)
    
class ReplayMemory:
    def __init__(self,capacity):
        self.memory = deque(maxlen=capacity)
        
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQNAgent:
    def __init__(self,num_states,num_actions,hidden=64):
        self.policy_net = DQN(num_states,hidden,num_states)
        self.target_net = DQN(num_states,hidden,num_states)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optm.Adam(self.policy_net.parameters(),lr=0.001)
        self.memory = ReplayMemory(1000)
        self.num_actions = num_actions
        self.num_actions = num_states
        self.gamma =0.99
        self.batch_size =64
        self.sync_rate =100
        self.step_done=0
        
        
    def state_to_tensor(self,state):
        x=torch.zeros(self.num_states)
        x[state] =1
        return x.float()
    
    def select_action(self,state,epsilon):
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        
        else:
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state)
                return self.policy_net(state_tensor).argmax().item()
            
    
    def remeber(self,state,action,next_state,reward,done):
        self.memory.append((state,action,next_state,reward,done))
        
    
    def optimize(self):
        if len(self.memory) <self.batch_size:
            return
        
        batch =self.memory.sample(self.batch_size)
        states,actions,next_states,rewardsdones =zip(*batch)
        
        states = torch.stack([self.state_to_tensor(s) for s in states])
        next_states =torch.stack([self.state_to_tensor(s) for s in next_states])
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards,dtype=torch.float32)
        dones = torch.tensor(dones,dtype=torch.float32)
        
        q_values= self.policy_net(states).gather(1,actions.unsqueeze(1)).sequeeze()
        
        with torch.no_grad():
            max_next_q =self.target_net(next_states).max(1)[0]
            target_q = rewards +self.gamma * max_next_q * (1-dones)
            
            
        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

     
        self.steps_done += 1
        if self.steps_done % self.sync_rate == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def train_pth(episodes=2000, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
    env = gym.make("D:\Way to Denmark\Reinforcement-Learning-codes\driver.pth", map_name="4x4", is_slippery=False)
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    agent = DQNAgent(num_states, num_actions)
    epsilon = epsilon_start

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            
            action = agent.select_action(state, epsilon)

           
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

           
            agent.remember(state, action, next_state, reward, done)

           
            agent.optimize()

            state = next_state
            total_reward += reward

       
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        
        torch.save(agent.policy_net.state_dict(), "frozenlake_dqn.pth")

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}/{episodes}, Reward: {total_reward}, Epsilon: {epsilon:.3f}, weights saved.")

    env.close()
    print("Training finished. Final weights saved to frozenlake_dqn.pth")
    return agent


def evaluate_agent(agent, episodes=10):
    env = gym.make("D:\Way to Denmark\Reinforcement-Learning-codes\driver.pth", map_name="4x4", is_slippery=False, render_mode="human")
    total_rewards = 0
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, epsilon=0.0)  
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_rewards += reward
    env.close()
    avg_reward = total_rewards / episodes
    print(f"Average reward over {episodes} episodes: {avg_reward}")
            
    
    
        
    
    
    

