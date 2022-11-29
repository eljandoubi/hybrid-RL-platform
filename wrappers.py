import gym

class PlatformWrapper(gym.Wrapper):
    """
    Extract only the useful features
    """
    def __init__(self,env):
        
        super(PlatformWrapper, self).__init__(env)
        
        self.env.observation_space=self.env.observation_space[0]

    def reset(self):
        (state,_),*_ = self.env.reset()
        return state

    def step(self, action):
        (state,_), reward, done, *_ = self.env.step(action)
        return state, reward, done