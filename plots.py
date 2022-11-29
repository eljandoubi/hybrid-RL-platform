import matplotlib.pyplot as plt
import os


def plot(series,names,title="",path="",show=False):
    "plot curves"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i,serie in enumerate(series,1):
        plt.plot(range(1, len(serie)+1), serie,label=names[i])
    plt.xlabel(names[0])
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(path,title+'.png'))
    if show:
        plt.show()
    
    
    
def render_policy(env, agent=None, horizon=100):
    "visulize the policy"
    if agent is not None:
        agent.eval()
        agent.load()
    
    state = env.reset()

    for timestep in range(horizon):
        env.render()
        if agent is None:
            action = env.action_space.sample()  # take random actions
        else:
            action = agent.act(state,0.,False)
            
        next_state, reward, done = env.step(action)
        state = next_state.copy()
        
        if done:
            state = env.reset()
            
    env.close()