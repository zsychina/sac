import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from models import Policy, QValue
from utils import ReplayBuffer


class Agent:
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 action_bound,
                 target_entropy,
                 max_buffer_size=int(1e6),
                 batch_size=256,
                 actor_lr=3e-4,
                 critic_lr=3e-3,
                 alpha_lr=3e-4,
                 tau=5e-3,  # soft update
                 gamma=0.99,
                 device='cuda'):
        
        self.actor = Policy(state_dim,
                            hidden_dim,
                            action_dim,
                            action_bound,).to(device)
        
        self.critic_1 = QValue(state_dim,
                               hidden_dim,
                               action_dim,).to(device)

        self.critic_2 = QValue(state_dim,
                               hidden_dim,
                               action_dim,).to(device)
        
        self.target_critic_1 = QValue(state_dim,
                                      hidden_dim,
                                      action_dim,).to(device)

        self.target_critic_2 = QValue(state_dim,
                                      hidden_dim,
                                      action_dim,).to(device)
        
        self.buffer = ReplayBuffer(state_dim, action_dim, device=device, max_size=max_buffer_size)
        self.batch_size = batch_size
        
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optim = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optim = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        
        self.log_alpha = torch.tensor(np.log(1e-2), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optim = optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state = state.unsqueeze(0)
        # [1, state_dim]
        action = self.actor(state)[0]  # get mean action [1, action_dim]
        action = action.squeeze(0)
        return action.cpu().detach().numpy()
    
    
    def cal_target(self, rewards, next_states, dones):
        next_actions, log_probs = self.actor(next_states)
        entropies = -log_probs.mean(dim=-1)
        entropies = entropies.unsqueeze(-1)
        
        q1_values = self.target_critic_1(next_states, next_actions)
        q2_values = self.target_critic_2(next_states, next_actions)
        
        next_values = torch.min(q1_values, q2_values) + self.log_alpha.exp() * entropies
        td_target = rewards + self.gamma * next_values * (1 - dones)
        return td_target
    
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        
    
    def update(self):
        states, actions, next_states, rewards, dones = self.buffer.sample(self.batch_size)

        # update Q
        td_target = self.cal_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach())
        )
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach())
        )        
        
        self.critic_1_optim.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optim.step()
    
        self.critic_2_optim.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optim.step()
        
        # update policy
        new_actions, log_probs = self.actor(states)
        entropies = -log_probs.mean(dim=-1)
        q1_values = self.critic_1(states, new_actions)
        q2_values = self.critic_2(states, new_actions)
        actor_loss = torch.mean(
            -self.log_alpha.exp() * entropies - torch.min(q1_values, q2_values)
        )
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        
        # update alpha
        alpha_loss = torch.mean(
            (entropies - self.target_entropy).detach() * self.log_alpha.exp()
        )
        self.log_alpha_optim.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optim.step()
        
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        
        
    def save_policy(self):
        torch.save(self.actor.state_dict(), './ckpt/actor.pt')    
      
    
    def load_policy(self):
        self.actor.load_state_dict(torch.load('./ckpt/actor.pt'))
        
        
        