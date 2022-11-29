import torch
import torch.nn as nn

class Scale(nn.Module):
    def __init__(self,param_min, param_max):
        super(Scale, self).__init__()
        self.a=0.5*(torch.tensor(param_max)-torch.tensor(param_min))
        self.b=0.5*(torch.tensor(param_max)+torch.tensor(param_min))
        self.scale=lambda x: self.a.to(x.device)*x+self.b.to(x.device)
        
    def __repr__(self):
        
        return self.__class__.__name__ +"(slope="+str(self.a)+", intercept="+str(self.b)+")"
        
    def forward(self,x):
        return self.scale(x)
    
class MH_net(nn.Module):
    
    def eval_model(self,*args):
        
        self.eval()
        with torch.no_grad(): 
            output=self.forward(*args)
        self.train()
        
        return output

class MHActor(MH_net):
    
    def __init__(self,state_size, action_param_dims,action_param_mins,
                 action_param_maxs, hidden_sizes, seed):
        super(MHActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        hidden_sizes_=hidden_sizes[:]
        
        hidden_sizes_.insert(0,state_size)
        
        self.body=nn.Sequential(*[nn.Sequential(nn.Linear(hidden_sizes_[i], hidden_sizes_[i+1]),
                                    nn.ReLU()) for i in range(len(hidden_sizes_)-1)])
        
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(hidden_sizes_[-1], dim),
                                    nn.Tanh(), Scale(param_min, param_max)) 
                                  for dim, param_min, param_max in 
                                  zip(action_param_dims,action_param_mins,action_param_maxs)])
        
    def forward(self,state):
        
        return tuple(head(self.body(state)) for head in self.heads)
    
    
    
class MHCritic(MH_net):
    def __init__(self,state_size, action_param_dims, hidden_sizes, seed):
        super(MHCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        hidden_sizes_=hidden_sizes[:]
        
        self.params_range = range(len(action_param_dims))
        
        self.feet=nn.ModuleList([nn.Sequential(nn.Linear(dim+state_size,hidden_sizes_[0]),
                                    nn.ReLU()) for dim in action_param_dims])
        
        self.body=nn.Sequential(*[nn.Sequential(nn.Linear(hidden_sizes_[i], hidden_sizes_[i+1]),
                                    nn.ReLU()) for i in range(len(hidden_sizes_)-1)])
        
        self.heads=nn.ModuleList([nn.Linear(hidden_sizes_[-1],1) for _ in self.params_range])
        
        
        
    def forward(self,state,params):
        
        output=[]
        
        for i in self.params_range:
            
            output.append(
                self.heads[i](
                    self.body(
                        self.feet[i](
                            torch.cat(
                                [state,params[i]]
                                ,dim=-1)
                            )
                        )
                    )
                )
        
        
        return torch.cat(output,dim=-1)
    
