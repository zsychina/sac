import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal


class Policy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(Policy, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound
        
    def forward(self, x):
        x = self.mlp(x)
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        action = torch.tanh(normal_sample)
        log_prob = dist.log_prob(normal_sample)
        
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob
    
    
class QValue(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValue, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)    
        return self.mlp(state_action)
    
    
    
    
    
        
if __name__ == '__main__':
    policy = Policy(10, 256, 3, 1)
    s = torch.rand(1, 10)
    print(policy(s))

    q = QValue(10, 256, 3)
    a = torch.rand(1, 3)
    print(q(s, a))
