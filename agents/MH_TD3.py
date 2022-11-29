from agents.BasicAgent import Agent
from agents.utils.OUNoise import OUNoise
from agents.models.MH_net import MHActor,MHCritic
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class MH_TD3(Agent):
    
    def __init__(self, state_size, action_param_dims, action_param_mins, 
                 action_param_maxs, buffer_size=int(1e4), batch_size=128,
                 random_seed=123, hidden_sizes_actor=[128,64], lr_actor=1e-4,
                 tau_actor=0.001, hidden_sizes_critic=[128, 64], lr_critic=1e-3,
                 tau_critic=0.1, grad_max=10., device="cpu", update_every=1,
                 gamma=0.99, path="saved\\", OUN=True, d=2):
        
        """Initialize an MP_DQN Agent object.
        
        Params
        ======
            state_size (int): state dimension
            action_param_sizes (list[int]): dimension of each action parameter
            action_param_mins (list[float]): min of each action parameter
            action_param_maxs (list[float]): max of each action parameter
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            random_seed (int): random seed
            hidden_sizes_actor (list[int]): list of hidden layers sizes of the actor network
            lr_actor (float): the actor leraning rate
            tau_actor (float): the target actor update rate
            hidden_sizes_critic (list[int]): list of hidden layers sizes of the critic network
            lr_critic (float): the critic leraning rate 
            tau_actor (float): the target critic update rate
            grad_max (float): max norm of the gradients
            device (str or torch.device): cpu/cuda
            update_every (int): update frequency
            gamma (float): discount factor
            path (str): path to save/load models
            OUN (bool): if True we use Ornstein-uhlenbeck noise else uniform sampling
            d (int): update frequency of the actor and the targets
        """
        
        super(MH_TD3,self).__init__(state_size, action_param_dims, action_param_mins,
                                    action_param_maxs, buffer_size, batch_size, 
                                    random_seed, device, update_every, gamma, path)
        
        self.memory.action_space="hybrid"
        
        self.grad_max=grad_max
        
        # Actor Network (w/ Target Network)
        self.actor_local = MHActor(state_size, action_param_dims, 
                                   action_param_mins, action_param_maxs,
                                   hidden_sizes_actor, random_seed).to(device)
        
        self.actor_target = MHActor(state_size, action_param_dims, 
                                    action_param_mins, action_param_maxs,
                                   hidden_sizes_actor, random_seed).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        self.tau_actor=tau_actor
        

        # Critic Network (w/ Target Network)
        self.critic_local_1 = MHCritic(state_size, action_param_dims, 
                                   hidden_sizes_critic, random_seed).to(device)
        
        self.critic_target_1 = MHCritic(state_size, action_param_dims, 
                                   hidden_sizes_critic, random_seed).to(device)
        
        self.critic_optimizer_1 = optim.Adam(self.critic_local_1.parameters(), lr=lr_critic)
        
        self.tau_critic=tau_critic
        

            
        self.critic_local_2 = MHCritic(state_size, action_param_dims, 
                                       hidden_sizes_critic, random_seed).to(device)
            
        self.critic_target_2 = MHCritic(state_size, action_param_dims, 
                                       hidden_sizes_critic, random_seed).to(device)
            
        self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(), lr=lr_critic)
            
        
        # Noise process
        self.OUN = OUN
        
        if OUN:
            
            self.noises = [OUNoise(s, random_seed) for s in action_param_dims]
            
        self.np_seed=np.random.seed(random_seed)
        
        self.d=d
        
        

    def reset(self):
        if self.OUN:
            for i in range(self.params_size):
                self.noises[i].reset()       
                    
                    
    def learn(self,experiences):
        """Update value parameters using given batch of experience tuples.
        Param
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        
        states, actions, rewards, next_states, dones = experiences
        
        
        # ---------------------------- update critic ---------------------------- #
        with torch.no_grad():
        
            # Get predicted next-state actions and Q values from target models
            next_params = self.actor_target(next_states)
        
                
            Q_targets_next_1 = self.critic_target_1(next_states, next_params
                                                    ).max(1)[0].unsqueeze(1)
                
            Q_targets_next_2 = self.critic_target_2(next_states, next_params
                                                    ).max(1)[0].unsqueeze(1)
                
            Q_targets_next=torch.minimum(Q_targets_next_1,Q_targets_next_2)
                
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
            
        # Compute critic loss 1
        Q_expected_1 = self.critic_local_1(states, actions["params"]
                                       ).gather(1, actions["actions"])
        
        critic_loss_1 = F.mse_loss(Q_expected_1, Q_targets)
        # Minimize the loss 1
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local_1.parameters(), self.grad_max)
        self.critic_optimizer_1.step()
        
        # Compute critic loss 2
        Q_expected_2 = self.critic_local_2(states, actions["params"]
                                       ).gather(1, actions["actions"])
        
        critic_loss_2 = F.mse_loss(Q_expected_2, Q_targets)
        # Minimize the loss 1
        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local_1.parameters(), self.grad_max)
        self.critic_optimizer_2.step()
        
        
        
        if self.t_step%self.d == 0:
        
            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            params_pred = self.actor_local(states)
            actor_loss = -self.critic_local_1(states, params_pred).mean()
            
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.grad_max)
            self.actor_optimizer.step()
            
            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local_1, self.critic_target_1, self.tau_critic)
            self.soft_update(self.critic_local_2, self.critic_target_2, self.tau_critic)
            self.soft_update(self.actor_local, self.actor_target, self.tau_actor)
        
    
    
    def act(self, state, eps=0., add_noise=False):
       """Returns action for given state as per current policy.
       
       Params
       ======
           state (array_like): current state
           eps (float): epsilon, for epsilon-greedy action selection
           add_noise (bool): if True, then add noise to actions parameters
       """
       
       state = torch.from_numpy(state).float().to(self.memory.device)
       
       action=list()
       
           
       if np.random.rand()<eps:
           action.append(np.random.randint(self.params_size))
           
           if self.OUN:
               
               
               params = self.actor_local.eval_model(state)
                   
               action.append(self.np_params(params,add_noise))
               
           else:
               params=tuple(np.random.uniform(
                   self.action_param_mins[i],
                   self.action_param_maxs[i],
                   self.action_param_dims[i])
                   for i in range(self.params_size))
                   
               action.append(params)

       else: 
           params = self.actor_local.eval_model(state)
           
           action_values = self.critic_local_1.eval_model(state,params)
           
           action.append(np.argmax(action_values.cpu().data.numpy()).astype(np.int32))
           
           action.append(self.np_params(params,add_noise*self.OUN))
           
           
       return tuple(action)
    
    def np_params(self,params,add_noise=False):
        """ Convert params to np.array """
            
        return tuple(np.clip(params[i].cpu().data.numpy()
                        +self.noises[i].sample()*add_noise,
                        self.action_param_mins[i],
                        self.action_param_maxs[i])
                for i in  range(self.params_size))
    
    
    def save(self,path=None):
        "Save each netorwk"
        if path is not None:
            self.path=path
            
        torch.save(self.actor_local.state_dict(), 
                   os.path.join(self.path,'checkpoint_actor_local_TD3.pth'))
        torch.save(self.critic_local_1.state_dict(), 
                   os.path.join(self.path,'checkpoint_critic_local_TD3_1.pth'))
        torch.save(self.critic_local_2.state_dict(), 
                   os.path.join(self.path,'checkpoint_critic_local_TD3_2.pth'))
        
    def load(self,path=None):
        "Load each netorwk"
        if self.path is None:
            return
            
        self.actor_local.load_state_dict(torch.load(
            os.path.join(self.path,'checkpoint_actor_local_TD3.pth')))
        self.critic_local_1.load_state_dict(torch.load(
            os.path.join(self.path,'checkpoint_critic_local_TD3_1.pth')))
        self.critic_local_2.load_state_dict(torch.load(
            os.path.join(self.path,'checkpoint_critic_local_TD3_2.pth')))
        
        
        
    def train(self):
        "train mode"
        self.actor_local.train()
        self.critic_local_1.train()
        self.critic_local_2.train()
        
    def eval(self):
        "eval mode"
        self.actor_local.eval()
        self.critic_local_1.eval()
        self.critic_local_2.eval()
        