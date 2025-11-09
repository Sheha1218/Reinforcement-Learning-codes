import gymnasium as gym
import numpy as np
import time 
import random
import tensorflow as tf
import rl
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


env = gym.make("CartPole-v1", render_mode="human")
states = env.observation_space.shape[0]
actions = env.action_space.n


episodes = 10
for episode in range(episodes):
    state = env.reset()[0]
    terminated = False
    truncated = False
    score =0
    
    
    while not (terminated or truncated ):
        env.render()
        action = random.choice([0,1])
        n_state, reward,terminated,truncated,info = env.step(action)
        score += reward
    
    print("Episode:{} Score: {}".format(episode+1,score))
    
    

def build_model(states,actions):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape =( 1,states)))
    model.add(tf.keras.layers.Dense(24,activation = 'relu'))
    model.add(tf.keras.layers.Dense(24,activation = 'relu'))
    model.add(tf.keras.layers.Dense(actions,activation = 'linear'))
    return model


model = build_model(states,actions)


def buid_agent(model,actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 50000, window_length=1)
    dpq =DQNAgent(model=model,memory =memory ,policy=policy,
                  nb_actions=actions,nb_steps_warmup=10,target_model_update=1e-2)

    return dpq


dpq = buid_agent(model,actions)
dpq.compile(tf.keras.optimizers.Adam(learning_rate=1e-3),metrics=['mae'])
dpq.fit(env,nb_steps=50000,visualize=False,verbose=1)

    
    
dpq.save_weights('dpq_weights.h5f',overwrite=True)
del model
del dpq
del env
env = gym.make('CartPole-v0')
actions = env.action_space.n
states = env.observation_space.shape[0]
model = build_model(states, actions)
model = build_model(states,action)
dpq.compile(tf.keras.optimizers.Adam(lr=1e-3,metrics=['mae']))

dpq.load_weights('dpq_weights.h5f')