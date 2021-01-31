import random
import gym
import numpy as np

action_space = [
    np.array([-0.8, 0.2, 0]), np.array([0.8, 0.2, 0]), np.array([-0.2, 0.8, 0]) , np.array([0.2, 0.8, 0]), 
    np.array([0, 1, 0]), np.array([0, 0.5, 0]), np.array([-1, 0, 0]), np.array([1, 0, 0]), 
    np.array([-1, 0, 0.3]), np.array([-1, 0, 0.8]), np.array([1, 0, 0.3]), np.array([1, 0, 0.8]), 
    np.array([-0.5, 0, 0.3]), np.array([-0.5, 0, 0.8]), np.array([0.5, 0, 0.3]), np.array([0.5, 0, 0.8])
]


class CarRacing:
    
    def __init__(self):
        
        self.env = gym.make('CarRacing-v1')
        self.env.action_space = action_space
        self.action_space = action_space
        self.observation_space = self.env.observation_space
        self.nb_offroad = 0
        self.offroad_tolerance = 5
        self.offroad_penalty = -100
        
        
    def reset(self):
        obs = self.env.reset()
        self.nb_offroad = 0
        return obs
    
    
    def step(self, action):
        # DQN syntax
        obs, reward, done, info = self.env.step(self.action_space[action])
        
        # Base syntax
        #obs, reward, done, info = self.env.step(action)
        
        # Case offroad (or maybe we didn't move)
        if reward < 0:
            self.nb_offroad += 1
            
        # Case we overcome tolerance, apply penalty 
        if self.nb_offroad >= self.offroad_tolerance:
            reward += self.offroad_penalty
            self.nb_offroad = 0
            
        return obs, reward, done, info
    
    
    def get_state(self):
        return deepcopy(self.env)
    
    
    def set_state(self, state):
        
        self.env = deepcopy(state)
        obs = np.array(list(self.env.unwrapped.state))
        return obs
    
    
    def render(self, mode='human'):
        return self.env.render(mode)
      
    def close(self):
        self.env.close()
        
        
        
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, repetitions):
        self.action_space = action_space
        self.repetitions = repetitions
        self.repeat = repetitions
        self.action = random.choice(self.action_space)

    def act(self, observation, reward, done):
        if self.repeat <= 0:
            self.repeat = self.repetitions
            self.action = random.choice(self.action_space)
            
        else:
            self.repeat -= 1
            
        return self.action
    
    
if __name__ == "__main__":
    
    from gym.envs.registration import registry, register, make, spec
    
    register(
        id='CarRacing-v1',
        entry_point='gym.envs.box2d:CarRacing',
        max_episode_steps=2000,
        reward_threshold=900,
    )
    
    env = CarRacing()
    agent = RandomAgent(env.action_space, 3)
    
    
    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            env.render()
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
                
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
    