import gym
import gym_platform
from wrappers import PlatformWrapper
from arguments import args
from agents.MH_TD3 import MH_TD3
from agents.MH_DQN import MH_DQN
from trainer import trainer
from tester import tester
from plots import plot,render_policy
from numpy import mean

if __name__=="__main__":

    env = gym.make('Platform-v0', apply_api_compatibility=True, disable_env_checker=True)
    env = PlatformWrapper(env)
    
    
    action_space=env.action_space[1]
    
    state_size=env.observation_space.shape[0]
    action_param_dims=[sa.shape[0] for sa in action_space]
    action_param_mins= [sa.low.item() for sa in action_space]
    action_param_maxs =[sa.high.item() for sa in action_space]
    
    
    trainer_vars=trainer.__code__.co_varnames
    
    agent_args=args.__dict__.copy()
    
    print("task : ",agent_args.pop('task'))
    
    del agent_args['agent_name']
    
    n_episodes_test=agent_args.pop("n_episodes_test")
    
    train_args={key:agent_args.pop(key) for key in trainer_vars if key in agent_args}
    
    test_args={key:value for key, value in train_args.items() if "eps" not in key}
    test_args["n_episodes"]=n_episodes_test
    
    
    
    agent=eval(args.agent_name)(state_size, action_param_dims, action_param_mins, 
                 action_param_maxs, **agent_args)
    
    print(agent)
    
    
    if args.task in ["all","tain"]:
        print("trainer :", train_args)
        curves_train=trainer(env,agent,**train_args)
        plot(curves_train,["Episode #","Score","Average Score"],"train",agent.path)
    
    if args.task in ["all","test"]:
        print("tester :", test_args)
        curves_test=tester(env,agent,**test_args)
        plot(curves_test,["Episode #","Score","Average Score"],"test",agent.path)
        print("Test mean reward: ", mean(curves_test[0]))
    
    if args.task in ["all","visu"]:
        render_policy(env,agent)
        

