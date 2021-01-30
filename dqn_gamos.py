import numpy as np

import gym
from gym.envs.registration import registry, register, make, spec

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from car_racing_custom import *


INPUT_SHAPE = (96, 96)
WINDOW_LENGTH = 3

register(
    id='CarRacing-v1',
    entry_point='gym.envs.box2d:CarRacing',
    max_episode_steps=2000,
    reward_threshold=900,
)
    
ENV_NAME = 'CarRacing-v1'
    
# Get the environment and extract the number of actions.
env = CarRacing()
np.random.seed(123)
env.seed(123)
print("nb actions = ", env.action_space)
nb_actions = len(env.action_space)
print("observation_space.shape = ", env.observation_space.shape)
input_shape = env.observation_space.shape

'''def build_model(input_shape, num_actions):
    input = Input(shape=(input_shape))
    x = Flatten()(input)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(num_actions, activation='linear')(x)
    model = tf.keras.Model(inputs=input, outputs=output)
    print(model.summary())
    return model'''

def build_model(input_space, nb_actions):
    model = Sequential()    
    model.add(Dense(24, activation='relu', input_shape=input_space))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())
    return model

model = build_model(input_shape, nb_actions)

'''model = Sequential()
model.add(Convolution2D(32, (8, 8), input_shape=input_shape))
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
print(model.summary())'''

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

    
memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=10000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               nb_steps_warmup=10, target_model_update=0.01)

dqn.compile(Adam(lr=.001), metrics=['mae'])


if __name__ == "__main__":
    
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
    weights_filename = 'dqn_' + ENV_NAME + '_weights.h5f'
    checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
    log_filename = 'dqn_' + ENV_NAME + '_log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=50000, visualize=False, verbose=2)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
