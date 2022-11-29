import numpy as np
from collections import deque


def trainer(env,agent, n_episodes=20000, max_t=1000, maxlen=1000, eps_start=1.0,
            eps_end=0.0, eps_decay=0.999, use_gamma=False):
    
    """function train an agent on an environnement.
    
    Params
    ======
        env (gym.env): environnement
        agent (Agent)
        n_episodes (int): number of episode to train on
        max_t (int): number of instant per episode
        maxlen (int): max number of episode to computer the average
        eps_start (float): initial epsilon
        eps_end (flaot): minmum epsilon
        eps_decay (flaot): decrease rate
        use_gamma (bool): if True evaluate episode score unsing agent.gamma
                            else gamma=1
    """
    
    agent.train()
    scores_deque = deque(maxlen=maxlen)
    scores = list()
    scores_mean = list()
    max_score = -np.Inf
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()                  
        agent.reset()                       # reset random process if any
        score_episode = list()
        for t in range(max_t):
            action=agent.act(state,eps,True)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
                
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
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        print('\rEpisode {}\tAverage Score: {:.4f}\tScore: {:.4f}'.format(i_episode, mean , score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, mean))
        if  mean>max_score:
            max_score=mean
            agent.save()
            
    return scores,scores_mean