import gymnasium as gym
import random
import numpy as np
import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy,EpsGreedyQPolicy

env = gym.make("ALE/SpaceInvaders-v3", render_mode="human")
height, width, channels = env.observation_space.shape
actions = env.action_space.n

episodes =5

for episode in range(1,episodes+1):
    state = env.reset()
    terminate = False
    truncated = False
    score =0
    
    while not (terminate or truncated):
        env.render()
        action = random.choice([0,1,2,3,4,5])
        n_state, reward,terminate,truncated,info = env.step(action)
        score += reward
env.close()


def buidl_model(height,width,channels,actions):
    model =tf.keras.models.Sequential()
    model =tf.keras.layers.Conv2D(32,(8,8),strides=(4,4),activation='relu',input_shape=(3,height,width,channels))
    model = tf.keras.layers.Conv2D(64,(4,4),strides=(2,2),activation='relu')
    model = tf.keras.layers.Conv2D(64,(3,3),activation='relu')
    model=tf.keras.layers.Flatten()
    model =tf.keras.layers.Dense(512,activation='relu')
    model =tf.keras.layers.Dense(256,activation='relu')
    model =tf.keras.layers.Dense(actions,activation='linear')
    
    return model




model = buidl_model(height,width,channels,actions)

def build_agent(model,actions):
    policy= LinearAnnealedPolicy(EpsGreedyQPolicy(),attr='eps',value_max=1.0,value_min=0.1,nb_steps=10000)
    memory = SequentialMemory(limit=1000,window_length=3)
    dqn = DQNAgent(model=model,memory=memory,policy=policy,enable_dueling_network=True,dueling_type='avg',
                   nb_actions=actions,nb_steps_warmup=10000)
    
    return dqn


dqn = build_agent(model,actions)
dqn.compile(tf.keras.optimizers.Adam(lr=1e-4))
dqn.fit(env,nb_steps=10000,visualize=False,verbose=2)



dqn.save_weights('dqn_weights.h5f')
del model,dqn
dqn.load_weights('dqn_weights.h5f')



