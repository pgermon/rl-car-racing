import os
import numpy as np

import gym
from gym.envs.registration import registry, register, make, spec

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input, Reshape
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from car_racing_v1 import *

def build_model(input_shape, nb_actions):
    model = Sequential()
    print(input_shape)
    
    model.add(Reshape(input_shape, input_shape = (1,96,96,3)))
    model.add(Convolution2D(32, (8, 8)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    
    print(model.summary())
    return model

def build_agent(model, nb_actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=10000)
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=nb_actions, nb_steps_warmup=500, target_model_update=100)
    return dqn

if __name__ == "__main__":

    INPUT_SHAPE = (96, 96, 3)

    register(
        id='CarRacing-v1',
        entry_point='gym.envs.box2d:CarRacing',
        max_episode_steps=2000,
        reward_threshold=900,
    )


    # Get the environment and extract the number of actions.
    ENV_NAME = 'CarRacing-v1'
    env = CarRacing()

    nb_actions = len(env.action_space)
    input_shape = env.observation_space.shape
    print("nb actions = ", len(env.action_space))
    print("observation_space.shape = ", input_shape)

    model = build_model(input_shape, nb_actions)

    dqn = build_agent(model, nb_actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    weights_filename = 'dqn_' + ENV_NAME + '_weights.h5f'
    checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
    log_filename = 'dqn_' + ENV_NAME + '_log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=2500)]
    callbacks += [FileLogger(log_filename, interval=100)]

    if os.path.exists(weights_filename + '.data-00000-of-00001') and os.path.exists(weights_filename + '.index'):
        dqn.load_weights(weights_filename)
        print('Weights loaded')
    else:
        print('Weights not found')

    # Train the agent
    dqn.fit(env, nb_steps=3000, verbose=2, nb_max_episode_steps=200, action_repetition=3, visualize = False)

     # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    NB_ITERATIONS = 1
    for i in range(NB_ITERATIONS):
      dqn.load_weights(weights_filename)
      dqn.fit(env, nb_steps=3000, verbose=2, nb_max_episode_steps=200, action_repetition=3, visualize = False)
      dqn.save_weights(weights_filename, overwrite=True)

    # Evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)



