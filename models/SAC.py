import numpy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import torch.optim as optim
import numpy as np

from . import utils_for_q_learning
from common.replay_buffer import ReplayBuffer

class SAC(nn.Module):
    def __init__(self, params, env, state_size, action_size, device):
        super(SAC, self).__init__()
        self.env = env
        self.device = device
        self.params = params
        self.max_a = self.env.action_space.high[0]

        # Defaults to False (standard RBFDQN behavior)

        self.buffer_object = ReplayBuffer(
            size=self.params['max_buffer_size']
        )

        self.state_size, self.action_size = state_size, action_size

        
        self.input_size = self.state_size
        
        self.policy_module = self._make_policy_module()
        self.value_module_1, self.value_module_2 = self._make_value_module()

        self.criterion = nn.MSELoss()
        
        self.policy_optimizer, self.value_optimizer_1, self.value_optimizer_2 = self._make_optimizers()
        
        self.num_updates = 0
        self.actor_update_delay = 2
        
        self.to(self.device)

    def _make_policy_module(self):
        module = nn.Sequential(
            nn.Linear(self.input_size, self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], 2*self.action_size),
        )
        
        return module
    
    def _make_value_module(self):
        module_1 = nn.Sequential(
            nn.Linear(self.input_size + self.action_size, self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], 1)
        )
        module_2 = nn.Sequential(
            nn.Linear(self.input_size + self.action_size, self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], 1)
        )
        
        return module_1, module_2
    
    def _make_optimizers(self):
        policy_optimizer = optim.Adam(self.policy_module.parameters(), lr=self.params["learning_rate_actor"])
        value_optimizer_1 = optim.Adam(self.value_module_1.parameters(), lr=self.params["learning_rate_critic"])
        value_optimizer_2 = optim.Adam(self.value_module_2.parameters(), lr=self.params["learning_rate_critic"])
        return policy_optimizer, value_optimizer_1, value_optimizer_2
    
    def _get_best_action(self, s):
        assert len(s.shape) == 2
        action_logits = self.policy_module(s) # shape [batch, 2]
        mu, log_sigma = action_logits[:, :self.action_size], action_logits[:, self.action_size:]
        sigma = torch.exp(log_sigma)
        covariance = (torch.eye(6).unsqueeze(0).to(self.device) * sigma)
        dist = MultivariateNormal(mu, covariance)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        action = torch.tanh(z)
        
        return action*self.max_a, log_prob
    
    def _get_sa_value(self, s, a):
        assert len(s.shape) == len(s.shape) == 2
        sa_concat = torch.cat([s, a], dim=1) # shape [batch_size, state_size + action_size]
        values_1 = self.value_module_1(sa_concat)
        values_2 = self.value_module_2(sa_concat)
        return values_1, values_2
    
    def get_best_qvalue_and_Action(self, s):
        best_action = self._get_best_action(s)[0]
        
        values_1, values_2 = self._get_sa_value(s, best_action)
        values = torch.min(torch.cat((values_1, values_2), dim=1), dim=1)[0]
        return values, best_action
    
    def forward(self, s, a):
        return self._get_sa_value(s, a)
    
    def enact_policy(self, s, episod, train_or_test, policy_type=None):
        self.eval()
        s_matrix = np.array(s).reshape(1, -1)
        # print(s)
        with torch.no_grad():
            s = torch.from_numpy(s_matrix).float().to(self.device)
            _, a = self.get_best_qvalue_and_Action(s)
            a = a.cpu().numpy()
        
        return a[0]
    
    def update(self, target_Q, sync_networks=True):
        if len(self.buffer_object) < self.params['batch_size']:
            return {"loss":0, "average_q":0, "average_next_q_max":0}
        s_matrix, a_matrix, r_matrix, sp_matrix, done_matrix, = self.buffer_object.sample(self.params['batch_size'])
        r_matrix = numpy.clip(r_matrix, a_min=-self.params['reward_clip'], a_max=self.params['reward_clip'])
        s_matrix = torch.from_numpy(s_matrix).float().to(self.device)
        a_matrix = torch.from_numpy(a_matrix).float().to(self.device)
        r_matrix = torch.from_numpy(r_matrix).float().to(self.device)
        done_matrix = torch.from_numpy(done_matrix).float().to(self.device)
        sp_matrix = torch.from_numpy(sp_matrix).float().to(self.device)
        
        with torch.no_grad():
            action_, log_prob_ = self._get_best_action(sp_matrix)
            values_1_, values_2_ = target_Q._get_sa_value(sp_matrix, action_)
            Q_star = torch.min(torch.cat((values_1_, values_2_), dim=1), dim=1)[0]
            y = r_matrix + (self.params['gamma'] * (1 - done_matrix) * (Q_star.squeeze() - self.params["entropy_weight"]*log_prob_))
            
        
        y_hat_1, y_hat_2 = self.forward(s_matrix, a_matrix)
        
        loss = self.criterion(y_hat_1.squeeze(), y) + self.criterion(y_hat_2.squeeze(), y)
        self.zero_grad()
        loss.backward()
        
        self.value_optimizer_1.step()
        self.value_optimizer_2.step()
        
        if self.num_updates % self.actor_update_delay == 0:
            action, log_prob = self._get_best_action(s_matrix)
            values_1, values_2 = self._get_sa_value(s_matrix, action)
            y_hat = torch.min(torch.cat((values_1, values_2), dim=1), dim=1)[0] - self.params["entropy_weight"]*log_prob
            neg_y_hat = -1*(y_hat)
            
            neg_y_hat_mean = neg_y_hat.mean()
            neg_y_hat_mean.backward()
            self.policy_optimizer.step()
            self.zero_grad()

            if sync_networks:
                utils_for_q_learning.sync_networks(
                    target=target_Q,
                    online=self,
                    alpha=self.params['target_network_learning_rate'],
                    copy=False)
                
        self.num_updates += 1
        
        loss = loss.item()
        average_q = y_hat_1.mean().item()
        average_next_q_max = Q_star.mean().item()
        return {
            "loss": loss,
            "average_q": average_q,
            "average_next_q_max": average_next_q_max
        }
            
            
            
        