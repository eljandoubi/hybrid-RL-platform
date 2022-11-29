import numpy as np
from collections import deque


def tester(env,agent,n_episodes=5000, max_t=1000, maxlen=1000, use_gamma=False):
    
    """function train an agent on an environnement.
    
    Params
    ======
        env (gym.env): environnement
        agent (Agent)
        n_episodes (int): number of episode to test on
        max_t (int): number of instant per episode
        maxlen (int): max number of episode to computer the average
        use_gamma (bool): if True evaluate episode score unsing agent.gamma
                            else gamma=1
    """
    
    agent.eval()
    agent.load()
    scores_deque = deque(maxlen=maxlen)
    scores = list()
    scores_mean = list()
    
    for i_episode in range(1, n_episodes+1):
        
        state = env.reset()                  
        agent.reset()                       # reset random process if any
        score_episode = list()
        
        for t in range(max_t):
            
            action=agent.act(state,0.,False)
            next_state, reward, done = env.step(action)
            state = next_state.copy()
            
            score_episode=np.append(score_episode,reward) 
                     
            if done:
                break 
        
        if use_gamma:
            score_episode*=agent.gamma**np.arange(len(score_episode))
            
        score=score_episode.sum()
        scores_deque.append(score)
        scores.append(score)
        mean=np.mean(scores_deque)
        scores_mean.append(mean)
        
        print('\rEpisode {}\tAverage Score: {:.4f}\tScore: {:.4f}'.format(i_episode, mean , score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, mean))
            
    return scores,scores_mean