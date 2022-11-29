import argparse
import os


parser = argparse.ArgumentParser(description='Train, test or visualize an agent to solve the platform environment')

parser.add_argument("--task", type=str,default="all",choices=["all","train","test","visu"])


parser.add_argument("--agent_name",type=str,choices=["MH_DQN","MH_TD3"], default="MH_DQN")

parser.add_argument("--buffer_size", type=int, default=10000, 
                    help="maximum size of buffer")

parser.add_argument("--batch_size", type=int, default=128, 
                    help="size of each training batch")

parser.add_argument("--random_seed", type=int, default=128)

parser.add_argument("--hidden_sizes_actor", type=int, default=[128, 64], nargs='*', 
                    help="list of hidden layers sizes of the actor network")

parser.add_argument("--lr_actor", type=float, default=1e-4, 
                    help="the actor leraning rate")

parser.add_argument("--tau_actor", type=float, default=0.001, 
                    help="the target actor update rate")

parser.add_argument("--hidden_sizes_critic", type=int, default=[128, 64], nargs='*', 
                    help="list of hidden layers sizes of the critic network")

parser.add_argument("--lr_critic", type=float, default=1e-3, 
                    help="the critic leraning rate")

parser.add_argument("--tau_critic", type=float, default=0.1, 
                    help="the target critic update rate")

parser.add_argument("--grad_max", type=float, default=10., 
                    help="max norm of the gradients")

parser.add_argument("--device",type=str,choices=["cpu","cuda"],default="cuda")


parser.add_argument("--update_every", type=int, default=1, 
                    help="update frequency")

parser.add_argument("--gamma", type=float, default=0.99, 
                    help="discount factor")

parser.add_argument("--path",type=str,default=None,
                    help="path to save/load models")

parser.add_argument("--OUN",type=bool,default=True,
                    help="if True we use Ornstein-uhlenbeck noise else uniform sampling")

parser.add_argument("--d", type=int, default=16, 
                    help="update frequency of the actor and the targets")

parser.add_argument("--n_episodes", type=int, default=20000,
                    help="number of episode to train on")

parser.add_argument("--max_t", type=int, default=1000,
                    help="number of instant per episode")

parser.add_argument("--maxlen", type=int, default=1000,
                    help="max number of episode to computer the average")

parser.add_argument("--eps_start", type=float, default=1.,
                    help="initial epsilon")

parser.add_argument("--eps_end", type=float, default=0.01,
                    help="minmum epsilon")
parser.add_argument("--eps_decay", type=float, default=0.995,
                    help="decrease rate")
parser.add_argument("--use_gamma",type=bool,default=False,
                    help="if True evaluate episode score unsing agent.gamma else gamma=1")

parser.add_argument("--n_episodes_test", type=int, default=5000,
                    help="number of episode to test on")


args=parser.parse_args()

if args.path is None:
    args.path=os.path.join("saved",args.agent_name)

