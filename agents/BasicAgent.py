from agents.utils.ReplayBuffer import ReplayBuffer
import os

class Agent():
    
    def __init__(self, state_size, action_param_dims, action_param_mins, action_param_maxs,
                 buffer_size, batch_size, random_seed, device, update_every, gamma, path):
       """Initialize an Agent object.
       
       Params
       ======
           state_size (int): state dimension
           action_param_sizes (list[int]): dimension of each action parameter
           action_param_mins (list[int]): min of each action parameter
           action_param_maxs (list[int]): max of each action parameter
           buffer_size (int): maximum size of buffer
           batch_size (int): size of each training batch
           random_seed (int): random seed
           device (str or torch.device): cpu/cuda
           update_every (int): update frequency
           gamma (float): discount factor
           path (str): path to save/load models
       """
       
       
       self.state_size = state_size
       
       self.action_param_dims = action_param_dims
       
       self.params_size = len(action_param_dims)
       
       self.action_param_mins = action_param_mins
       self.action_param_maxs = action_param_maxs
       

       # Replay memory
       self.memory = ReplayBuffer(buffer_size, batch_size, random_seed, device, self.params_size)
       
       # Initialize time step (for updating every UPDATE_EVERY steps)
       self.update_every = update_every
       self.t_step = 0
       
       
       self.random_seed = random_seed
       
       self.gamma = gamma
       
       path=os.path.join(os.getcwd(),path)
       
       if not os.path.isdir(path):
           os.makedirs(path)
       
       self.path=path
       
       
       
    def reset(self):
        pass
    
    def __repr__(self):
        ch=self.__class__.__name__+"(\n"
        for key,arg in self.__dict__.items():
            ch+="\t"+key+"="+str(arg)+"\n"
        ch+=")\n\n"
        return ch
       
       
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) >= self.memory.batch_size:
        
            if self.t_step == 0:
                
                experiences = self.memory.sample()
                self.learn(experiences)
                
                
    def learn(self,experiences):
        """Update value parameters using given batch of experience tuples.
        Param
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        raise NotImplementedError
        
        
    def act(self, state, eps=0., add_noise=False):
       """Returns action for given state as per current policy.
       
       Params
       ======
           state (array_like): current state
           eps (float): epsilon, for epsilon-greedy action selection
           add_noise (bool): if True, then add noise to actions parameters
       """
       raise NotImplementedError
       
            

    def soft_update(self, local_model, target_model,tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
    def save(self,path=None):
        "Save each netorwk"
        raise NotImplementedError
        
        
    def load(self):
        "Load each netorwk"
        raise NotImplementedError
        
    def train(self):
        "train mode"
        raise NotImplementedError
    
    def eval(self):
        "eval mode"
        raise NotImplementedError
    
    
    